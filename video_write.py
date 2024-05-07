from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import os

def write_video(image_files, fps, name = os.path.join(os.path.abspath("output/"), 'video_enhanced.mp4')):
    clip = ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(name, fps=fps, logger='bar', codec='libx265')
    print(f"Video saved to: {name}")

if __name__ == '__main__':
    rgb_files = [os.path.abspath('enhance/'+x) for x in os.listdir(os.path.abspath('enhance/')) if x.startswith('rgb_')]
    d_files = [os.path.abspath('enhance/'+x) for x in os.listdir(os.path.abspath('enhance/')) if x.startswith('d_')]
    write_video(rgb_files, 25*2, name = os.path.join(os.path.abspath('output'), 'rgb_enhanced.mp4'))
    write_video(d_files, 25*2, name = os.path.join(os.path.abspath('output'), 'depth_enhanced.mp4'))