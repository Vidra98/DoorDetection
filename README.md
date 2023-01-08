# PyTorch-Pose

PyTorch-Pose is a PyTorch implementation of the general pipeline for 2D single human pose estimation. The aim is to provide the interface of the training/inference/evaluation, and the dataloader with various data augmentation options for the most popular human pose databases (e.g., [the MPII human pose](http://human-pose.mpi-inf.mpg.de), [LSP](http://www.comp.leeds.ac.uk/mat4saj/lsp.html) and [FLIC](http://bensapp.github.io/flic-dataset.html)).

Some codes for data preparation and augmentation are brought from the [Stacked hourglass network](https://github.com/anewell/pose-hg-train). Thanks to the original author.

This code has been adapted to do door detection in the Super Proton Synchrotron accelerator

## Features
- Multi-thread data loading
- Multi-GPU training
- Logger
- Training/testing results visualization

## Installation
1. PyTorch (>= 0.2.0): Please follow the [installation instruction of PyTorch](http://pytorch.org/). Note that the code is developed with Python2 and has not been tested with Python3 yet.

2. Clone the repository with submodule
   ```
   git clone --recursive https://github.com/Vidra98/DoorDetection
   ```

4. Modify your `.bashrc` file:
   ```
   export PYTHONPATH=".:$PYTHONPATH"
   ```

## Usage

In our case, we want to identify the doors keypoints to do pose estimation later on 

### Dataset 
To be able to train the model, the first step is to annotate the dataset. This is done using CVAT and custom video of the gate in the mockup and later on in the tunnel.

The point are always annotated in the order : bottom left -> top left -> top right -> bottom right
### Folder architecture

### Training
Run the following command in terminal to train an 1-stack of hourglass network on the door dataset.
**a modifier**
```
CUDA_VISIBLE_DEVICES=0 python example/mpii.py -a hg --stacks 1 --blocks 1 --checkpoint checkpoint/mpii/hg8 -j 4
```
**a modifier**

Here,
* `CUDA_VISIBLE_DEVICES=0` identifies the GPU devices you want to use. For example, use `CUDA_VISIBLE_DEVICES=0,1` if you want to use two GPUs with ID `0` and `1`.
* `-j` specifies how many workers you want to use for data loading.
* `--checkpoint` specifies where you want to save the models, the log and the predictions to.

Please refer to the `example/**a modifier**.py` for the supported options/arguments.

### Testing

To test the network you can run :

For a quick start, you can retrain the network using the command 

```
```

```
CUDA_VISIBLE_DEVICES=0 python example/mpii.py -a hg --stacks 2 --blocks 1 --checkpoint checkpoint/mpii/hg_s2_b1 --resume checkpoint/mpii/hg_s2_b1/model_best.pth.tar -e -d
```

* `-a` specifies a network architecture
* `--resume` will load the weight from a specific model
* `-e` stands for evaluation only
* `-d` will visualize the network output. It can be also used during training
* EXPLAIN OTHER HYPER PARAMETER

The result will be saved as a `.mat` file (`preds_valid.mat`), which is a `2958x16x2` matrix, in the folder specified by `--checkpoint`.

#### Result

## Contribute
Please create a pull request if you want to contribute.
