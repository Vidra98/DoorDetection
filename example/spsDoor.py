from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt
import glob

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, door_final_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
import pose.datasets as datasets
import numpy as np

import cv2

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

idx = [1,2,3,4]

best_acc = 0

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    global best_acc

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks, num_blocks=args.blocks, 
                                    num_classes=args.num_classes, model_inplane=args.model_inplane)

    model = torch.nn.DataParallel(model).to(device)
    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss(reduction='mean').to(device)

    optimizer = torch.optim.RMSprop(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #define the input and output resolution of the model
    out_res = args.output_res
    inp_res = 2 * out_res

    # optionally resume from a checkpoint
    title = 'spsDoor-' + args.arch
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(device))

            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    cudnn.benchmark = True  # There is BN issue here see https://github.com/bearpaw/pytorch-pose/issues/33
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        datasets.spsDoor('data/spsDoor/train2/annotations/instances_default.json', 
                        'data/spsDoor/train2/images', train_data = True, 
                        sigma=args.sigma, label_type=args.label_type,
                        inp_res=inp_res, out_res=out_res, data_aug=args.data_aug),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.spsDoor('data/spsDoor/test/annotations/instances_default.json', 
                        'data/spsDoor/test/images', val_data = True,
                        sigma=args.sigma, label_type=args.label_type, train=False,
                        inp_res=inp_res, out_res=out_res, data_aug=args.data_aug),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.spsDoor('data/spsDoor/val/annotations/instances_default.json', 
                        'data/spsDoor/val/images', val_data = True,
                        sigma=args.sigma, label_type=args.label_type, train=False,
                        inp_res=inp_res, out_res=out_res, data_aug=args.data_aug),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        print('\nEvaluation only')
        loss, acc, predictions = validate(val_loader, model, criterion, args.num_classes, 
                                            args.debug, args.flip, out_res)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    if args.predict:
        print('\Prediction only')
        predictions = predict(val_loader, model, criterion, args.num_classes, 
                                            args.debug, args.flip, out_res)
        return

    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *=  args.sigma_decay
            val_loader.dataset.sigma *=  args.sigma_decay
        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, 
                                    optimizer, args.debug, args.flip, out_res)
        # evaluate on test set
        valid_loss, valid_acc, predictions = validate(test_loader, model, criterion, args.num_classes,
                                                      args.debug, args.flip, out_res)

        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        is_best = train_acc > best_acc -0.00001
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, predictions, is_best, checkpoint=args.checkpoint)
        torch.save(model.state_dict(), os.path.join(args.checkpoint, "model_dict.ckpt"))


    logger.close()
    logger.plot(['Train Acc', 'Val Acc'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(train_loader, model, criterion, optimizer, debug=False, flip=False, out_res=64):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    gt_win, pred_win = None, None
    bar = Bar('Processing', max=len(train_loader))

    for i, (input, target, meta) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)
        input, target = input.to(device), target.to(device, non_blocking=True)
        # compute output
        output = model(input)

        loss = criterion(output[0], target)
        for j in range(1, len(output)):
            loss += criterion(output[j], target)
        acc = accuracy(output[0], target, idx)

        if debug: # visualize groundtruth and predictions
            gt_batch_img = batch_with_heatmap(input, target)
            pred_batch_img = batch_with_heatmap(input, output[0])
            if not gt_win or not pred_win:
                ax1 = plt.subplot(121)
                ax1.title.set_text('Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                ax2 = plt.subplot(122)
                ax2.title.set_text('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acces.update(acc[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=acces.avg
                    )
        bar.next()

    bar.finish()
    return losses.avg, acces.avg


def validate(val_loader, model, criterion, num_classes, debug=False, flip=False, out_res=64):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)
    ground_truth = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for i, (input, target, meta) in enumerate(val_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device, non_blocking=True)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        output = model(input)
        
        # max = np.max(output[0][0][1].cpu().detach().numpy())
        # min = np.min(output[0][0][1].cpu().detach().numpy())
        # im=np.array((output[0][0][1].cpu().detach().numpy()-min)/(max-min))
        # print(max-min)
        # plt.figure()
        # plt.imshow(im)
        # plt.show()
        # cv2.destroyAllWindows()

        score_map = output[-1].cpu()
        if flip:
            flip_input = torch.autograd.Variable(
                    torch.from_numpy(fliplr(input.clone().numpy())).float().to(device),
                    volatile=True
                )
            flip_output_var = model(flip_input)
            flip_output = flip_back(flip_output_var[-1].cpu())
            score_map += flip_output



        loss = 0
        for o in output:
            loss += criterion(o, target)
        acc = accuracy(score_map, target.cpu(), idx)
        
        # generate predictions
        gt = door_final_preds(target.cpu(), [meta['width'], meta['height']], [out_res, out_res])
        preds = door_final_preds(score_map, [meta['width'], meta['height']], [out_res, out_res])
        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]
            ground_truth[meta['index'][n], :, :] = gt[n, :, :]

        if debug:
            gt_batch_img = batch_with_heatmap(input, target)
            pred_batch_img = batch_with_heatmap(input, score_map)

            if not gt_win or not pred_win:
                ax1 = plt.subplot(121)
                ax1.title.set_text('Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                ax2 = plt.subplot(122)
                ax2.title.set_text('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.draw()
            #plt.waitforbuttonpress()
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acces.update(acc[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    data=data_time.val,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=acces.avg
                    )
        bar.next()

    if debug:
        images_folder = os.getcwd()+os.sep+"data/spsDoor/val/images/"
        for i in range(len(os.listdir(images_folder))):
            img_path=images_folder + "door_" + str(i+1) + ".jpg"
            img=cv2.imread(img_path)
            img_pred = predictions.numpy()[i]
            img_gt = ground_truth.numpy()[i]
            for point in img_pred :
                img = cv2.circle(img, tuple([int(i) for i in point]), 5, (0,255,255), 2)
            for point in img_gt :
                img = cv2.circle(img, tuple([int(i) for i in point]), 5, (0,0,255), 2)
            cv2.imshow(img_path, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    
    bar.finish()
    return losses.avg, acces.avg, predictions

def predict(val_loader, model, criterion, num_classes, debug=False, flip=False, out_res=64):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    model.eval()

    pred_win = None
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for i, (input, target, meta) in enumerate(val_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device, non_blocking=True)

        # compute output
        output = model(input)
        score_map = output[-1].cpu()

        loss = 0
        
        # generate predictions
        preds = door_final_preds(score_map, [meta['width'], meta['height']], [out_res, out_res])
        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]

        if debug:
            pred_batch_img = batch_with_heatmap(input, score_map)

            if not pred_win:
                plt.figure()
                plt.title('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                pred_win.set_data(pred_batch_img)
            plt.waitforbuttonpress()
            plt.draw()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
 
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    data=data_time.val,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    )
        bar.next()

    if debug:
        images_folder = os.getcwd()+os.sep+"data/spsDoor/val/images/"
        for i in range(len(os.listdir(images_folder))):
            img_path=images_folder + "door_" + str(i) + ".jpg"
            img=cv2.imread(img_path)
            img_pred = predictions.numpy()[i]
            for point in img_pred :
                img = cv2.circle(img, tuple([int(i) for i in point]), 5, (0,255,255), 2)
            cv2.imshow(img_path, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    
    bar.finish()
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-s', '--stacks', default=1, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=512, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=4, type=int, metavar='N',
                        help='Number of keypoints')
    parser.add_argument('--model_inplane', default=64, type=int, metavar='N',
                        help='Number of keypoints')
    parser.add_argument('--output_res', type=int, default=256,
                        help='resolution of the output of the model : [output_res, output_res]')
    # Training strategy
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=2, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[160, 290],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Data processing
    parser.add_argument('-aug', '--augment', dest='data_aug', action='store_true', default=False,
                        help='flip the input during validation')
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-p', '--predict', dest='predict', action='store_true',
                        help='predict model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')


    main(parser.parse_args())
