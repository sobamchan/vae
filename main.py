import torch
import argparse
from trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def main():
    args = get_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
