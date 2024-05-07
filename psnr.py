#from TorchDB import DBreader_frame_interpolation
from data import Test_DBreader_frame_interpolation
from torch.utils.data import DataLoader
from model import SepConvNet
import argparse
import torch
from torch.autograd import Variable
import gc

parser = argparse.ArgumentParser(description='SepConv Pytorch')

# parameters
parser.add_argument('--psnr_files', type=str, default='./psnr')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--load_model', type=str, default=None)


@torch.no_grad()
def main():
    """
    The main function parses the command line arguments, loads the data and the model, 
    and calculates the PSNR (Peak Signal-to-Noise Ratio) for frames in the dataset.
    """
    args = parser.parse_args()
    db_dir = args.psnr_files

    batch_size = args.batch_size

    datareader = Test_DBreader_frame_interpolation(db_dir, psnr_calc=True)
    train_loader = DataLoader(dataset=datareader, batch_size=batch_size, shuffle=False,\
                                num_workers=1, persistent_workers=True, prefetch_factor=1, drop_last=True, pin_memory=False)


    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        kernel_size = checkpoint['kernel_size']
        model = SepConvNet(kernel_size=kernel_size)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.epoch = checkpoint['epoch']
    else:
        raise Exception("Checkpoint required for PSNR calculation")

    model = model.cuda()


    # model.eval()
    # TestDB.Test(model, test_output_dir, logfile, str(model.epoch.item()).zfill(3) + '.png')

    print("Starting psnr")
    psnr_list = []
    psnr_rgb_list = []
    for frame0, frame1, frame2 in train_loader:
        batch_psnr = model.calculate_psnr(frame0, frame2, frame1)
        batch_psnr_rgb = model.calculate_psnr_rgb(frame0, frame2, frame1)
        if batch_psnr != float('inf'):
            psnr_list.append(batch_psnr)
        if batch_psnr_rgb != float('inf'):
            psnr_rgb_list.append(batch_psnr_rgb)
        del frame0, frame1, frame2
        gc.collect()
        torch.cuda.empty_cache()
    print('{:<15s}{:<20.16f}'.format('PSNR: ', sum(psnr_list)/len(psnr_list)))
    print('{:<15s}{:<20.16f}'.format('PSNR RGB: ', sum(psnr_rgb_list)/len(psnr_rgb_list)))
    
    


if __name__ == "__main__":
    main()
