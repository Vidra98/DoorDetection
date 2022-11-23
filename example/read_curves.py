from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt
import glob
import scipy.io
import shutil
from pose.utils.logger import Logger, savefig
import numpy as np
import pose.models as models
import pandas as pd

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main(args):
    try:
        df = pd.read_csv(args.path+os.sep+'log.csv')
    except:
        try :
            df = pd.read_csv(args.path+os.sep+'log.txt', sep='\t')
        except:
            print('No csv nor txt files detected')
            return False
        

    print(df.keys)
    epoch = df['Epoch'].values
    tr_loss = df['Train Loss'].values
    val_loss = df['Val Loss'].values
    tr_acc = df['Train Acc'].values
    val_acc = df['Val Acc'].values

    plot_figure(epoch, tr_loss, 'loss', 'Epoch', 'Training loss', ['training', 'validation'])
    plot_figure(epoch, val_loss, 'loss', 'Epoch', 'Validation loss', ['training', 'validation'])
    plt.savefig(args.path+os.sep+'loss.png')
    plot_figure(epoch, tr_acc, 'accuracy', 'Epoch', 'Training accuracy', ['Training', 'Validation'])
    plot_figure(epoch, val_acc, 'accuracy', 'Epoch', 'Validation accuracy', ['Training', 'Validation'])
    plt.savefig(args.path+os.sep+'accuracy.png')
    plt.show()

    return True

def plot_figure(x_data, y_data, title='', xlabels='', ylabel='', legend=''):
    plt.figure(title)
    plt.plot(x_data, y_data)
    plt.title(title)
    plt.xlabel(xlabels)
    plt.ylabel(ylabel)
    plt.legend(legend)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training loss and accuracy curves')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-c', '--path', default='', type=str, metavar='PATH',
                        help='path to save model file (default: \'\')')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    main(parser.parse_args())

