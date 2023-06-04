import argparse
import torch
from PIL import Image
from torchvision import transforms
import os
import time
import subprocess

def main(args):
    # Load the TorchScript model
    model = torch.jit.load('20x/modelpnnx20x.pt')

    # Move the model to the device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device
    print(f'Using device: {device}')


    # Directory path of input frames
    frames_dir = args.frames_dir

    # Output directory path to save generated frames
    output_dir = 'output_frames'
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

        image_a = Image.open(frame_a_path)
        image_b = Image.open(frame_b_path)

        # Resize the images
        image_a = image_a.resize(desired_size)
        image_b = image_b.resize(desired_size)

        # Convert the images to tensors
        transform = transforms.ToTensor()
        tensor_a = transform(image_a)
        tensor_b = transform(image_b)

        # Move the tensors to the device
        tensor_a = tensor_a.to(device)
        tensor_b = tensor_b.to(device)

        # Run inference
        output_tuple = model(tensor_a.unsqueeze(0), tensor_b.unsqueeze(0))
        output = output_tuple[0]  # Access the desired tensor from the tuple

        # Rescale the output tensor to the range [0, 1]
        output = (output - output.min()) / (output.max() - output.min())

        # Squeeze the tensor to remove the extra dimensions
        output = output.squeeze()

        # Convert the output tensor to a PIL image
        output_image = transforms.ToPILImage()(output.cpu())

        # Save the output image
        output_path = os.path.join(output_dir, f'output_{i + 1:03d}.png')
        output_image.save(output_path)

        if (i + 1) % 10 == 0:
            print(f'Processed {i + 1} frames out of {num_frames - 1}')

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, required=True, help='Path to the input frames directory')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cpu', 'cuda', 'mps', 'xla', 'tpu', 'vulkan'],
                        help='select device [auto/cpu/cuda/mps/xla/tpu/vulkan] (default: auto)')
    args = parser.parse_args()

    main(args)
