from config import parse_args
import models
from load_data import FoodDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from tqdm import tqdm
import sys


def train(args, device, train_loader, model, criterion, optimizer, scheduler):
    since = time.time()

    for epoch in range(args.N_epochs):
        print('Epoch {}/{}'.format(epoch+1, args.N_epochs))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        for ob_rgb, ob_depth, local_rgb, local_depth, ob_ee, interact_rgb, interact_depth, interact_ee, action in tqdm(train_loader):

            # to device
            ob_rgb = ob_rgb.to(device, dtype=torch.float32)
            ob_depth = ob_depth.to(device, dtype=torch.float32)
            local_rgb = local_rgb.to(device, dtype=torch.float32)
            local_depth = local_depth.to(device, dtype=torch.float32)
            ob_ee = ob_ee.to(device, dtype=torch.float32)
            interact_rgb = interact_rgb.to(device, dtype=torch.float32)
            interact_depth = interact_depth.to(device, dtype=torch.float32)
            interact_ee = interact_ee.to(device, dtype=torch.float32)
            action = action.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs =  model((ob_rgb, ob_depth, local_rgb, local_depth, ob_ee, interact_rgb, interact_depth, interact_ee))
            loss = criterion(outputs, action)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * ob_rgb.size(0)
        
        scheduler.step(epoch)
        epoch_loss = running_loss/len(train_loader.dataset)

        print("lr: {}".format(scheduler.optimizer.param_groups[0]['lr']))
        print("train Loss: {:.9f}".format(epoch_loss))
        print("")
        if (epoch+1)%args.save_epoch==0:
            torch.save(model, 'epoch_{}.pth'.format(epoch+1))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # save model
    torch.save(model, 'model.pth')

class Print_Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    sys.stdout = Print_Logger("train_log.txt")
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare dataset
    train_set = FoodDataset(args)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # setting for training training
    model = models.SCONE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0, eps=1e-07, amsgrad=False)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, min_lr=0.00001)

    # train
    train(args, device, train_loader, model, criterion, optimizer, scheduler)

if __name__ == '__main__':
    main()