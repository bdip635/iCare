import numpy as np
import argparse
import os, sys
import torch

argv = sys.argv[1:]
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    prog='PROG',)
    
parser.add_argument('--learning_rate', '-r',
                    type=float,
                    default=1e-4,
                    help='Initial learning rate for adam')

parser.add_argument('--batch_size',
                    type=int,
                    default=16,
                    help='Batch size for training and validation')

parser.add_argument('--epochs', type=int, default=100,
                    help="Total number of training epochs")
                    

args = parser.parse_args(argv)
DEVICE= "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE= (512, 512)
LEARNING_RATE= args.learning_rate
NUM_EPOCHS= args.epochs
WEIGHTS_FILE= 'weights/wide_resnet50_2.pth.tar'
NUM_CLASSES=2
BATCH_SIZE= args.batch_size

if __name__=='__main__':
    print("Configurations set to:\n")
    print(args)
