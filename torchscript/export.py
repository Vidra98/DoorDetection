import argparse
import torch.utils
import torch
import pose.models as models
import cv2 as cv
from pose.utils.imutils import *
import matplotlib.pyplot as plt

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    
    # create model
    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))

    if args.export_only and args.inport_only:
        print("Nothing to be done there")
        return

    img_path="/home/victor/Desktop/python/deep_learning/pytorch-pose-7ce6642f777e9da6249bd5b05330d57fa09ea37a/data/spsDoor/train2/images/door_6.jpg"
    img = door_load_image(img_path, 512)  # CxHxW
    input = img.to(device, non_blocking=True)
    input.unsqueeze_(0)

    if args.inport_only is False:
        model = models.__dict__[args.arch](num_stacks=args.stacks, num_blocks=args.blocks, 
                                        num_classes=args.num_classes, model_inplane=args.model_inplane)

        checkpoint = torch.load(args.checkpoint + "model_best.pth.tar", map_location=torch.device(device))
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()

        example = torch.rand(1, 3, 512, 512)
        traced_script_module = torch.jit.trace(model, example, strict=False)
        traced_script_module.eval()
        traced_script_module.forward(example)
        traced_script_module.save(args.checkpoint + "traced_model.pt")

        output_1 = model(input)

        im_keypoint00=np.array(output_1[0][0][0].cpu().detach().numpy())
        im_keypoint01=np.array(output_1[0][0][1].cpu().detach().numpy())
        im_keypoint02=np.array(output_1[0][0][2].cpu().detach().numpy())
        im_keypoint03=np.array(output_1[0][0][3].cpu().detach().numpy())

        plt.figure("model")
        plt.subplot(221)
        plt.imshow(im_keypoint00)
        plt.subplot(222)
        plt.imshow(im_keypoint01)
        plt.subplot(223)
        plt.imshow(im_keypoint02)
        plt.subplot(224)
        plt.imshow(im_keypoint03)

    if args.export_only is False:

        model_loaded = torch.jit.load(args.checkpoint + "traced_model.pt")

        output_2 = model_loaded(input)
        
        keypoint0=np.array(output_2[0][0][0].cpu().detach().numpy())
        keypoint1=np.array(output_2[0][0][1].cpu().detach().numpy())
        keypoint2=np.array(output_2[0][0][2].cpu().detach().numpy())
        keypoint3=np.array(output_2[0][0][3].cpu().detach().numpy())

        plt.figure("reloaded model")
        plt.subplot(221)
        plt.imshow(keypoint0)
        plt.subplot(222)
        plt.imshow(keypoint1)
        plt.subplot(223)
        plt.imshow(keypoint2)
        plt.subplot(224)
        plt.imshow(keypoint3)
    
    
    plt.show()
    plt.waitforbuttonpress()
    print("saved and loaded successfully, torch version is {}, device is {}".format(torch.__version__, device))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stacks', default=1, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=4, type=int, metavar='N',
                        help='Number of keypoints')
    parser.add_argument('--model_inplane', default=64, type=int, metavar='N',
                        help='Number of keypoints')
    parser.add_argument('--checkpoint', type=str, metavar='PATH', required=True,
                        default="checkpoint/spsdoor/conv1_stride1/hg1b1_aug_data2/")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-e', '--export_only', dest='export_only', action='store_true', default=False,
                        help='flip the input during validation')
    parser.add_argument('-i', '--inport_only', dest='inport_only', action='store_true', default=False,
                        help='flip the input during validation')
    # Parse args
    args = parser.parse_args()
    main(args)

