# pytorch-sepconv-rgbd
This is a reference implementation of Video Frame Interpolation via Adaptive Separable Convolution [1] using PyTorch. Given two frames, it will make use of [adaptive convolution](http://graphics.cs.pdx.edu/project/adaconv) [2] in a separable manner to interpolate the intermediate frame. Should you be making use of the work, please cite the paper [1].

This is a modified version of [modified code](github.com/HyeongminLEE/pytorch-sepconv) which is in turn modified from this version [original code](https://github.com/sniklaus/pytorch-sepconv).

<a href="https://arxiv.org/abs/1708.01692" rel="Paper"><img src="http://content.sniklaus.com/sepconv/paper.jpg" alt="Paper" width="100%"></a>

## Difference from the original code
1. Modified model to include depth information.
2. Added reconstruction loss using VGG-19 model.
3. Changed dataset to load RGBD images.

## Setup
The separable convolution layer is implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided binary packages as outlined in the CuPy repository.
Install the dependencies
```
pip install -r requirements.txt
```

## To Prepare Training Dataset
The training dataset is not provided. We prepared training dataset by cropping [Free Viewpoint RGB-D Video Dataset](https://medialab.sjtu.edu.cn/post/free-viewpoint-rgb-d-video-dataset/). When creating training dataset, we measured Optical Flow of each patch to balance the motion magnitude of whole dataset.

## Train
```
python train.py --train ./inside/rgbd/dataset --out_dir ./output/folder/tobe/created --kernel <int_kernel_size> --epochs <num_epochs>
```

## Test
### Enhance video
```
python test_generate_interpolated_video.py --input ./test/input/of/rgbd/data --output ./output/folder/tobe/created --checkpoint ./dir/for/pytorch/checkpoint
```
### Generate frame output
```
python generate_frame_output.py --test_files ./test/input/of/rgbd/data --out_dir ./output/folder/tobe/created --load_model ./dir/for/pytorch/checkpoint --frame_idx <frame_number_pair_in_dataset>
```
### Calculate PSNR
```
python psnr.py --psnr_files ./test/input/of/rgbd/data --batch_size <low_batch_size> --load_model ./dir/for/pytorch/checkpoint
```

## video
<a href="https://cometmail-my.sharepoint.com/:v:/r/personal/sxa220028_utdallas_edu/Documents/vfi_rgbd_output.mp4?csf=1&web=1&e=KDF9sW&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D" rel="Video"><img src="" alt="Video (will expire in 2025)" width="100%"></a>

## License
The provided implementation is strictly for academic purposes only. Should you be interested in using our technology for any commercial use, please feel free to contact us.

## References
```
[1]  @inproceedings{Niklaus_ICCV_2017,
         author = {Simon Niklaus and Long Mai and Feng Liu},
         title = {Video Frame Interpolation via Adaptive Separable Convolution},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2017}
     }
```

```
[2]  @inproceedings{Niklaus_CVPR_2017,
         author = {Simon Niklaus and Long Mai and Feng Liu},
         title = {Video Frame Interpolation via Adaptive Convolution},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2017}
     }
```

## TODO
Due to time constraints, code refactoring, input arguments, and cleanup was not completed.
Non-exhaustive list of remaining work:
1. Modify Dataset to accept rgbd data in other formats. Currently it can only accept split depth and rgb videos with a specific data structures.
2. Extensive code documentation and improvement of this readme to include more optional arguments.
3. Find a way to write video directly without JPEG compression as an intermediate step as it reduces quality.