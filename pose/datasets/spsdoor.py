from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *
import albumentations as A
from matplotlib import pyplot as plt

class spsDoor(data.Dataset):
    def __init__(self, jsonfile, img_folder, train_data=False, test_data=False, val_data=False, inp_res=256, out_res=64, train=True, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian', data_aug=False, return_label=True):
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type
        self.data_augmentation = data_aug
        self.return_label=return_label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create train/val split
        with open(jsonfile) as anno_file:   
            self.anno = json.load(anno_file)
        self.train, self.valid, self.test = [], [], []
        for idx, val in enumerate(self.anno['annotations']):
            if val_data == True:
                self.valid.append(idx)
            elif test_data == True:
                self.test.append(idx)
            elif train_data == True:
                self.train.append(idx)
        self.mean, self.std = self._compute_mean()

        if self.data_augmentation:
            self.transform = A.Compose(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                    A.RGBShift(r_shift_limit=0.04, g_shift_limit=0.04, b_shift_limit=0.025, p=0.5),
                    A.GaussNoise(var_limit=(0.00025,0.00025),p=0.5),
                    A.CoarseDropout(max_holes=10, max_height=30,max_width = 30,p=1),
                    #A.ShiftScaleRotate(shift_limit=0.05,scale_limit=0.2, rotate_limit=4, p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.3)
                ],
                additional_targets={'mask0': 'mask', 'mask1': 'mask', 
                                    'mask2': 'mask', 'mask3': 'mask'}
            )

    def _compute_mean(self):
        meanstd_file = './data/spsDoor/mean.pth.tar'
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            print('==> compute mean')
            mean = torch.zeros(3)
            std = torch.zeros(3)
            cnt = 0
            for index in self.train:
                cnt += 1
                print( '{} | {}'.format(cnt, len(self.train)))
                im = self.anno['images'][index]
                img_path = os.path.join(self.img_folder, im['file_name'])
                img = load_image(img_path) # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
            
        return meanstd['mean'], meanstd['std']

    def _visualize(self, image):
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(image)

    def __getitem__(self, index):
        if self.is_train:
            im = self.anno['images'][self.train[index]]
            if self.return_label:
                annotation = self.anno['annotations'][self.train[index]]
        else:
            im = self.anno['images'][self.valid[index]]
            if self.return_label:
                annotation = self.anno['annotations'][self.valid[index]]

        img_path = os.path.join(self.img_folder, im['file_name'])
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        img = door_load_image(img_path, self.inp_res)  # CxHxW
        r = 0
        if self.is_train:
           
            # # Flip
            # if random.random() <= 0.5:
            #     img = torch.from_numpy(fliplr(img.numpy())).float()
            #     pts = shufflelr(pts, width=img.size(2), dataset='mpii')
            #     c[0] = img.size(2) - c[0]

            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        #inp = color_normalize(img, self.mean, self.std)
        inp=img
        # Generate ground truth
        if self.return_label:
            pts = torch.Tensor(annotation['segmentation'])
            tpts = pts.clone()
            nparts = int((pts.size(1)*pts.size(0))/2)
            target = torch.zeros(nparts, self.out_res, self.out_res)
            for i in range(nparts):
                if annotation['iscrowd'] == 0: # COCO visible: 0-no label, 1-label + invisible, 2-label + visible
                    for segment in tpts:
                        segment[2*i] = segment[2*i]/im['width']*self.out_res
                        segment[2*i+1] = segment[2*i+1]/im['height']*self.out_res
                        target[i] = draw_labelmap(target[i], segment[2*i:2*i+2]-1, \
                            self.sigma, type=self.label_type)

        #print('inp shape ', inp.shape, 'type ', inp.dtype, '\n', inp)
        #print('target shape ', target.shape, 'type ', target.dtype, '\n', target)

        if self.data_augmentation:
            transformed = self.transform(image=inp.to(self.device).numpy().transpose([1, 2, 0]), mask0=target[0].numpy(), mask1=target[1].numpy(),
                mask2=target[2].numpy(), mask3=target[3].numpy())

            #self._visualize(inp.numpy().transpose([1, 2, 0]))

            inp = torch.tensor(transformed['image'].transpose([2, 0, 1]))
            masks = np.array([transformed['mask0'], transformed['mask1'],
                                transformed['mask2'], transformed['mask3']])
            target = torch.tensor(masks)

            #self._visualize(transformed['image'])

            sum_mask=0.3*(target[0]+target[1]+target[2]+target[3])

            #self._visualize(sum_mask.numpy())
            #print('inp shape ', inp.shape, 'type ', inp.dtype, '\n', inp)
            #print('target shape ', target.shape, 'type ', target.dtype, '\n', target)
            #plt.show()
        
        # Meta info
        if self.return_label:
            meta = {'index' : index, 'pts' : pts, 'tpts' : tpts,
                'width' : im['width'], 'height' : im['height']}
            return inp, target, meta

        else:
            meta = {'index' : index, 'width' : im['width'], 'height' : im['height']}
            return inp, torch.empty(1), meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)