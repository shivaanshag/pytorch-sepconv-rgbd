from data import DBreader_frame_interpolation
from torch.utils.data import DataLoader
from model import SepConvNet
import argparse
import torch
from torch.autograd import Variable
import os
import time

parser = argparse.ArgumentParser(description='SepConv Pytorch')

# parameters
parser.add_argument('--train', type=str, default='./rgbd')
parser.add_argument('--kernel', type=int, default=51)
parser.add_argument('--sub_window_size', type=int, default=128)
parser.add_argument('--frame_limit_per_file', type=int, default=1001)
parser.add_argument('--train_num_files', type=int, default=6)
parser.add_argument('--out_dir', type=str, default='./output')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--load_model', type=str, default=None)


def main():
    args = parser.parse_args()
    db_dir = args.train

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    result_dir = args.out_dir + '/result'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    logfile = open(args.out_dir + '/log.txt', 'w')
    logfile.write('batch_size: ' + str(args.batch_size) + '\n')

    total_epoch = args.epochs
    batch_size = args.batch_size

    # Initialize training data
    datareader = DBreader_frame_interpolation(db_dir, args.sub_window_size, \
                                              args.frame_limit_per_file, args.train_num_files)
    train_loader = DataLoader(dataset=datareader, batch_size=batch_size, shuffle=True,\
                                num_workers=1, prefetch_factor=1, persistent_workers=True,\
                                    drop_last=True, pin_memory=False)


    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        kernel_size = checkpoint['kernel_size']
        model = SepConvNet(kernel_size=kernel_size)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.epoch = checkpoint['epoch']
    else:
        kernel_size = args.kernel
        model = SepConvNet(kernel_size=kernel_size)

    logfile.write('kernel_size: ' + str(kernel_size) + '\n')

    model = model.cuda()
    max_step = len(datareader)

    print("Starting training")
    start_time = time.time_ns()
    last_time = start_time
    while True:
        if model.epoch.item() == total_epoch:
            break
        model.increase_epoch()
        model.train()
        for i, (frame0, frame1, frame2) in enumerate(train_loader):
            loss = model.train_model(frame0, frame2, frame1)
            if i % 400 == 0:
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'\
                      .format('Train Epoch: ', '[' + str(model.epoch.item()) + \
                              '/' + str(total_epoch) + ']', 'Step: ', '[' + str(i+1) + \
                              '/' + str(max_step//batch_size) + ']', 'train loss: ', loss.item()))
        if model.epoch.item() % 1 == 0:
            torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), \
                        'kernel_size': kernel_size}, ckpt_dir + '/model_epoch' \
                            + str(model.epoch.item()).zfill(3) + '.pth')
            curr_time = time.time_ns()
            print('{:<13s}{:<14s}{:<6s}{:<12.16f}'\
                  .format('Time to train Epoch: ', '[' + str(model.epoch.item()) + \
                          '/' + str(total_epoch) + ']', 'time: ', (curr_time-last_time)/1e9))
            last_time = curr_time
        torch.cuda.empty_cache()
    print('{:<13s}{:<14s}{:<6s}{:<12.16f}'\
          .format('Epochs trained:', str(total_epoch), \
                  'Time elapsed (in minutes):',  (curr_time-start_time)/1e9/60))
    logfile.close()


if __name__ == "__main__":
    main()
