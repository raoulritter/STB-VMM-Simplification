import argparse
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from utils.data_loader import ImageFromFolderTest

def main(args):
    # Device choice (auto)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    print(f'Using device: {device}')

    # Load TorchScript model
    model = torch.jit.load(args.load_ckpt).to(device)
    print("=> loaded TorchScript model '{}'".format(args.load_ckpt))

    # Check saving directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)

    # Data loader
    dataset_mag = ImageFromFolderTest(
        args.video_path, mode=args.mode, num_data=args.num_data, preprocessing=False)
    data_loader = data.DataLoader(dataset_mag,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.workers,
                                  pin_memory=False)

    # Generate frames
    model.eval()

    # Modify your loop over the data_loader like this:
    for i, (xa, xb, _) in enumerate(data_loader):
        if i % args.print_freq == 0:
            print('processing sample: %d' % i)

        xa = xa.to(device)
        xb = xb.to(device)

        # Since the model expects a tuple, we pass the inputs as a tuple
        y_hat, _, _, _ = model(xa, xb)
        if i == 0:
            # Back to image scale (0-255)
            tmp = xa.permute(0, 2, 3, 1).cpu().detach().numpy()
            tmp = np.clip(tmp, -1.0, 1.0)
            tmp = ((tmp + 1.0) * 127.5).astype(np.uint8)

            # Save first frame
            fn = os.path.join(save_dir, 'STBVMM_%s_%06d.png' % (args.mode, i))
            im = Image.fromarray(np.concatenate(tmp, 0))
            im.save(fn)

        # back to image scale (0-255)
        y_hat = y_hat.permute(0, 2, 3, 1).cpu().detach().numpy()
        y_hat = np.clip(y_hat, -1.0, 1.0)
        y_hat = ((y_hat + 1.0) * 127.5).astype(np.uint8)

        # Save frames
        fn = os.path.join(save_dir, 'STBVMM_%s_%06d.png' % (args.mode, i+1))
        im = Image.fromarray(np.concatenate(y_hat, 0))
        im.save(fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Swin Transformer Based Video Motion Magnification')

    # Application parameters
    parser.add_argument('-i', '--video_path', type=str, metavar='PATH', required=True,
                        help='path to video input frames')
    parser.add_argument('-c', '--load_ckpt', type=str, metavar='PATH', required=True,
                        help='path to load checkpoint')
    parser.add_argument('-o', '--save_dir', default='demo', type=str, metavar='PATH',
                        help='path to save generated frames (default: demo)')
    parser.add_argument('--mode', default='static', type=str, choices=['static', 'dynamic'],
                        help='magnification mode (static, dynamic)')
    parser.add_argument('-n', '--num_data', type=int, metavar='N', required=True,
                        help='number of frames')

    # Execute parameters
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='batch size (default: 1)')
    parser.add_argument('-p', '--print_freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='select device [auto/cpu/cuda] (default: auto)')

    args = parser.parse_args()

    main(args)
