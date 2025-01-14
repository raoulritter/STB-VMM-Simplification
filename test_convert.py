import torch
from PIL import Image
from torchvision import transforms
import os
import time
import subprocess
import sys

def main():
    # Load the TorchScript model
    model = torch.jit.load('20x/modelpnnx20x.pt')

    # Move the model to the GPU (Metal)
    model = model.to('mps')

    # Directory path of input frames
    frames_dir = os.path.join('Original', sys.argv[1])

    # Output directory path to save generated frames
    output_dir = os.path.join('output_frames', sys.argv[1] + '_amplified')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Resize the images if necessary
    desired_size = (384, 384)

    # Get the list of frames in the directory
    frame_files = sorted(os.listdir(frames_dir))

    # Count the number of frames
    num_frames = len(frame_files)

    # Process the frames and generate output images
    start_time_frames = time.time()
    for i in range(num_frames - 1):
        # Load the frames
        frame_a_path = os.path.join(frames_dir, frame_files[i])
        frame_b_path = os.path.join(frames_dir, frame_files[i + 1])

        # [...]

    end_time_frames = time.time()
    frames_time = end_time_frames - start_time_frames
    print(f'Processing frames completed. Time taken: {frames_time:.2f} seconds')

    # Use ffmpeg to create a 30 fps video
    start_time_video = time.time()
    output_video_path = os.path.join(output_dir, 'output_video.mp4')
    subprocess.call(['ffmpeg', '-framerate', '30', '-i', os.path.join(output_dir, 'output_%03d.png'),
                     '-c:v', 'libx264', '-r', '30', '-pix_fmt', 'yuv420p', output_video_path])

    end_time_video = time.time()
    video_time = end_time_video - start_time_video
    total_time = frames_time + video_time
    print(f'Video created. Time taken: {video_time:.2f} seconds')
    print(f'Total time elapsed: {total_time:.2f} seconds')

    # Save the timing information to a text file
    with open(os.path.join(output_dir, 'timing_info.txt'), 'w') as f:
        f.write(f'Processing frames completed. Time taken: {frames_time:.2f} seconds\n')
        f.write(f'Video created. Time taken: {video_time:.2f} seconds\n')
        f.write(f'Total time elapsed: {total_time:.2f} seconds\n')


if __name__ == '__main__':
    main()
