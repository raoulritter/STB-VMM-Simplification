import argparse
import os
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms

def main(args):
    # Device choice (auto)
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f'Using device: {device}')

    # Load ONNX model
    onnx_session = ort.InferenceSession(args.load_ckpt)
    
    # Check saving directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)

    # Load and process the image
    image_path = args.video_path
    image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize((768, 384)),  # Resize to the size your model expects
        transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
    ])

    image = transform(image)
    image = image.unsqueeze(0)  # Add extra dimensions to match the model's expected input size
    input_array = image.cpu().numpy()  # Convert tensor to numpy array

    # Run inference
    y_hat = onnx_session.run(None, {onnx_session.get_inputs()[0].name: input_array})

    # Process output
    y_hat = np.array(y_hat)
    y_hat = np.clip(y_hat, -1.0, 1.0)
    y_hat = ((y_hat + 1.0) * 127.5).astype(np.uint8)

    # Save output
    fn = os.path.join(save_dir, 'STBVMM_%s_%06d.png' % (args.mode, 0))
    im = Image.fromarray(np.squeeze(y_hat))
    im.save(fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ONNX implementation of Swin Transformer Based Video Motion Magnification')

    # Application parameters
    parser.add_argument('-i', '--video_path', type=str, metavar='PATH', required=True,
                        help='path to input image')
    parser.add_argument('-c', '--load_ckpt', type=str, metavar='PATH', required=True,
                        help='path to load ONNX model')
    parser.add_argument('-o', '--save_dir', default='demo', type=str, metavar='PATH',
                        help='path to save output image (default: demo)')
    parser.add_argument('--mode', default='static', type=str, choices=['static', 'dynamic'],
                        help='magnification mode (static, dynamic)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='select device [auto/cpu/cuda] (default: auto)')

    args = parser.parse_args()

    main(args)
