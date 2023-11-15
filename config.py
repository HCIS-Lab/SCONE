import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", type=str, 
                        help="Folder containing dataset")
    parser.add_argument('--batch_size', default=128,
                        type=int, help='batch size')
    parser.add_argument('--num_workers', default=8,
                        type=int, help='num_workers in dataloader')
    # training
    parser.add_argument('--N_epochs', default=500, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--save_epoch', default=100, type=int,
                        help='each #epoch save model weights')
    args = parser.parse_args()
    return args
