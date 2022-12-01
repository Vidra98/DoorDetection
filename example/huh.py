
from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt

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
import matplotlib as mpl 



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main(args):
    inp_res = 256
    out_res = 520
    data=datasets.spsDoor('data/spsDoor/train/annotations/instances_default.json', 
                      'data/spsDoor/train/images', train_data=True,
                      sigma=args.sigma, label_type=args.label_type, train=True,
                      inp_res=inp_res, out_res=out_res)
    for image in range(1,15):
        inp, targets, meta = data[image]#args.image]
        img=inp.numpy().transpose(1,2,0)
        plt.figure()
        meta['width']= [meta['width']]
        meta['height']= [meta['height']]
        gt_targets = targets[None,:]
        gt = door_final_preds(gt_targets.cpu(), [meta['width'], meta['height']], [out_res, out_res])
        img = cv2.resize(img, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
        for point in gt[0] :
            print(point)
            img = cv2.circle(img, tuple([int(i) for i in point]), 5, (255,0,0), 2)
        plt.imshow(img)
        # plt.figure()
        # for idx, target in enumerate(targets):
        #     plt.subplot(2,2,idx+1)
        #     plt.imshow(target)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Data processing
    parser.add_argument('-im', '--image', type=int, default=0,
                        help='image to plot')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    main(parser.parse_args())