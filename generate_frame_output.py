from data import Test_DBreader_frame_interpolation
from torchvision.utils import save_image
from model import SepConvNet
import argparse
import torch
import gc
import os

parser = argparse.ArgumentParser(description='SepConv Pytorch')

# parameters
parser.add_argument('--test_files', type=str, default='./psnr')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--frame_idx', type=int, default=298)
parser.add_argument('--out_dir', type=str, default='./output/images')


@torch.no_grad()
def main():
    args = parser.parse_args()
    db_dir = args.test_files
    datareader = Test_DBreader_frame_interpolation(db_dir, psnr_calc=False)


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

    test_idx = args.frame_idx
    out_dir = args.out_dir
    frame0, frame2 = datareader[test_idx]

    epoch = str(checkpoint['epoch'].item())

    frame0 = torch.unsqueeze(frame0, 0)
    frame2 = torch.unsqueeze(frame2, 0)
    interpolated = model(frame0, frame2)
    save_image(frame0[0,:3], out_dir + os.pathsep + epoch + '_prev.png')
    save_image(interpolated[0,:3], out_dir + os.pathsep + epoch + '_interpolated.png')
    save_image(frame2[0,:3], out_dir + os.pathsep + epoch + '_next.png')

    del frame0, frame2
    gc.collect()
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    main()
