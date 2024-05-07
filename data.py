import numpy as np
from os import listdir
from os.path import join, isdir
from torch.utils.data import Dataset
import torch
from torchvision.transforms import v2
import cv2
from multiprocessing import Pool, Manager
from random import randint
import gc
from random import shuffle
import sys
from patch_selection import select_patches_from_frames



class DBreader_frame_interpolation(Dataset):
    """
    Database reader class creates triplets (prev, gt, next) for frame
    interpolation training
    """

    def __init__(self, db_dir, sub_window_size = 128, frame_limit_per_file = 1001, train_num_files = 6):
        self.device = 'cuda'
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop((sub_window_size,sub_window_size)),
        ])
        self.transform_gt = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop((sub_window_size,sub_window_size))
        ])


        # RGBD dataset specific file_list
        video_folder_list = np.array([(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))])
        file_list = [(vid + '/RGB/' + str(i)+'.mp4', vid + '/Depth/' + str(i)+'.mp4') for i in range(0,12) for vid in video_folder_list]
        
        manager = Manager()
        self.datapoints= manager.list()
        shuffle(file_list)
        print("num of video pairs found:", len(file_list))
        file_list = file_list[:min(train_num_files, len(file_list))]
        print("Selected", len(file_list), "video pairs from the above, processing...")
        for file_rgb, file_depth in file_list:

            rgbVideoFrames = cv2.VideoCapture(file_rgb)
            depthVideoFrames = cv2.VideoCapture(file_depth)
            rgb_frame_list = manager.list()
            depth_frame_list = manager.list()

            num_frames = 0
            while rgbVideoFrames.isOpened() and num_frames < frame_limit_per_file:
                ret, rgb = rgbVideoFrames.read()
                if not ret: break
                rgb_frame_list.append(rgb)
                num_frames += 1
            rgbVideoFrames.release()

            depth_frames = 0
            while depthVideoFrames.isOpened() and depth_frames < frame_limit_per_file:
                ret, depth = depthVideoFrames.read()
                if not ret: break
                depth_frame_list.append(cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY))
                depth_frames += 1
            depthVideoFrames.release()

            if len(rgb_frame_list) != len(depth_frame_list):
                print('Skipping dataset:', file_rgb.split('/')[2], file_rgb.split('/')[4], \
                  'as unequal frames in rgb and depth images', len(rgb_frame_list), 'vs', len(depth_frame_list), 'respectively')
                continue

            print('Reading frames complete for dataset:', file_rgb.split('/')[2], file_rgb.split('/')[4], "-", num_frames, 'frames with shape:', rgb_frame_list[0].shape)


            idx_list = [(i-2, i-1, i) for i in range(2, len(rgb_frame_list), 2)]
            with Pool() as pool:
                # Map the worker function to the inputs and the shared array
                pool.starmap(select_patches_from_frames, [(rgb_frame_list, depth_frame_list, *idxs, self.datapoints) for idxs in idx_list])
            
            print('Processing complete for', file_rgb.split('/')[2]+ ' - ' + file_rgb.split('/')[4])
            del rgb_frame_list
            del depth_frame_list
            gc.collect()

        self.datapoints = list(self.datapoints)
        print("num datapoints",len(self.datapoints))


    def __getitem__(self, index):
        frame0 = self.transform(self.datapoints[index][0])
        frame1 = self.transform_gt(self.datapoints[index][1])
        frame2 = self.transform(self.datapoints[index][2])
        
        # random order flip
        if randint(0,1): frame0, frame2 = frame2, frame0 
        
        if randint(0,1): # random flip vertical
            frame0 = torch.flip(frame0, [1])
            frame1 = torch.flip(frame1, [1])
            frame2 = torch.flip(frame2, [1])
        if randint(0,1): # random flip horizontal
            frame0 = torch.flip(frame0, [2])
            frame1 = torch.flip(frame1, [2])
            frame2 = torch.flip(frame2, [2])


        return frame0.to(device=self.device), frame1.to(device=self.device), frame2.to(device=self.device)


    def __len__(self):
        return len(self.datapoints)


class Test_DBreader_frame_interpolation(Dataset):
    """
    Database reader class creates triplets (prev, gt, next) for frame
    interpolation testing
    """

    def __init__(self, db_dir, psnr_calc = False, shuffle = False, test_frame_limit_per_file = 500, test_num_files = 1):
        self.psnr_calc = psnr_calc
        self.test_frame_limit_per_file = test_frame_limit_per_file
        self.device = 'cuda'
        self.transform = v2.Compose([v2.ToTensor()])

        # RGBD dataset specific file_list
        video_folder_list = np.array([(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))])
        file_list = [(vid + '/RGB/' + str(i)+'.mp4', vid + '/Depth/' + str(i)+'.mp4') for i in range(0,12) for vid in video_folder_list]

        self.datapoints= []
        if shuffle: shuffle(file_list)

        print("num of video pairs found:", len(file_list))
        file_list = file_list[:min(test_num_files, len(file_list))]
        print("Selected", len(file_list), "video pairs from the above, processing...")
        file_rgb, file_depth = file_list[0]
        rgbVideoFrames = cv2.VideoCapture(file_rgb)
        depthVideoFrames = cv2.VideoCapture(file_depth)
        rgb_frame_list = []
        depth_frame_list = []

        num_frames = 0
        while rgbVideoFrames.isOpened() and num_frames < self.test_frame_limit_per_file:
            ret, rgb = rgbVideoFrames.read()
            if not ret: break
            rgb_frame_list.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            num_frames += 1
        rgbVideoFrames.release()

        depth_frames = 0
        while depthVideoFrames.isOpened() and depth_frames < self.test_frame_limit_per_file:
            ret, depth = depthVideoFrames.read()
            if not ret: break
            depth_frame_list.append(cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY))
            depth_frames += 1
            # if num_frames == 1: print('test',cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY).shape)
        depthVideoFrames.release()

        if len(rgb_frame_list) != len(depth_frame_list):
            print('Skipping dataset:', file_rgb.split('/')[2], file_rgb.split('/')[4], \
                'as unequal frames in rgb and depth images', len(rgb_frame_list), 'vs', len(depth_frame_list), 'respectively')
            sys.exit()

        print('Reading frames complete for dataset:', file_rgb.split('/')[2], file_rgb.split('/')[4], "-", num_frames, 'frames with shape:', (rgb_frame_list[0].shape)[:2])
        # print(depth_frame_list[0].shape)
        if not psnr_calc:
            self.datapoints = [(torch.vstack((self.transform(rgb_frame_list[i-1]), self.transform(depth_frame_list[i-1]))),\
                                torch.vstack((self.transform(rgb_frame_list[i]), self.transform(depth_frame_list[i]))))\
                                    for i in range(1,len(rgb_frame_list))]
        else: # for psnr calculations
            self.datapoints = [(torch.vstack( [self.transform(rgb_frame_list[i-2]), self.transform(depth_frame_list[i-2])] ),
                                torch.vstack( [self.transform(rgb_frame_list[i-1]), self.transform(depth_frame_list[i-1])] ),\
                                torch.vstack( [self.transform(rgb_frame_list[i]  ), self.transform(depth_frame_list[i]  )] ) )\
                                    for i in range(2,len(rgb_frame_list),2) ]
        del rgb_frame_list

        del depth_frame_list 
        gc.collect()
        
        print('Processing complete for', file_rgb.split('/')[2]+ ' - ' +file_rgb.split('/')[4])
        print("num datapoints", len(self.datapoints))


    def __getitem__(self, index):
        if self.psnr_calc:
            frame0 = self.datapoints[index][0].to(device=self.device)
            frame1 = self.datapoints[index][1].to(device=self.device)
            frame2 = self.datapoints[index][2].to(device=self.device)
        else:
            frame0 = self.datapoints[index][0].to(device=self.device)
            frame2 = self.datapoints[index][1].to(device=self.device)

        return (frame0, frame1, frame2) if self.psnr_calc else (frame0, frame2)


    def __len__(self):
        return len(self.datapoints)

    
if __name__=='__main__':
    # test code for data.py
    test_loader = Test_DBreader_frame_interpolation('./test')
    b = test_loader[0]
    print(b[0])
