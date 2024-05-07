import argparse
from data import Test_DBreader_frame_interpolation
from model import SepConvNet
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import gc
from video_write import write_video

parser = argparse.ArgumentParser(description='SepConv Pytorch')

# parameters
parser.add_argument('--input', type=str, default='./test')
parser.add_argument('--output', type=str, default='./enhance')
parser.add_argument('--checkpoint', type=str, default='./output/checkpoint/model_epoch650.pth')

@torch.no_grad()
def main():
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    ckpt = args.checkpoint

    print("Reading Test DB...")
    test_datareader = Test_DBreader_frame_interpolation(input_dir)
    test_loader = DataLoader(dataset=test_datareader, batch_size=1, shuffle=False,\
                        num_workers=0, prefetch_factor=None, persistent_workers=False, drop_last=True, pin_memory=False)
    frame_count = len(test_datareader)
    print("Loading the Model...")
    checkpoint = torch.load(ckpt)
    kernel_size = checkpoint['kernel_size']
    model = SepConvNet(kernel_size=kernel_size)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.epoch = checkpoint['epoch']
    model.cuda()


    print("Test Start...")
    counter = 0
    rgb_file_list = [output_dir + f'/rgb_{0:05d}.jpeg']
    depth_file_list = [output_dir + f'/d_{0:05d}.jpeg']
    for i, frames in enumerate(test_loader):
        frame0 = frames[0]
        frame2 = frames[1]
        if i == 0:
            d_frame = frame0[0, 3:, :, :]
            save_image(frame0[0, :3, :, :], rgb_file_list[-1])
            save_image(d_frame, depth_file_list[-1])
        generated = model(frame0, frame2)
        nextframe = frame2

        counter += 1
        rgb_file_list += [output_dir + f'/rgb_{counter:05d}.jpeg', output_dir + f'/rgb_{counter+1:05d}.jpeg']
        save_image(generated[0, :3, :, :], rgb_file_list[-2])
        # save_image(frame0[0, :3, :, :], rgb_file_list[-2])
        save_image(nextframe[0, :3, :, :], rgb_file_list[-1])

        depth_file_list += [output_dir + f'/d_{counter:05d}.jpeg', output_dir + f'/d_{counter+1:05d}.jpeg']
        save_image(generated[0, 3:, :, :], depth_file_list[-2])
        # save_image(frame0[0, 3:, :, :], depth_file_list[-2])
        save_image(nextframe[0, 3:, :, :], depth_file_list[-1])
        print(f"\033[KProgress: [{'='*round((counter/frame_count)*50):<100}] ({counter//2}/{frame_count})", end='\r')
        counter += 1

        del nextframe
        gc.collect()
        torch.cuda.empty_cache()


    write_video(rgb_file_list, 25*2, 'enhanced_rgb.mp4')
    write_video(depth_file_list, 25*2, 'enhanced_depth.mp4')


if __name__ == "__main__":
    main()
