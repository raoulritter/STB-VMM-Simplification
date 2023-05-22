import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
from PIL import Image

from utils.ImageFromFolderDouble import ImageFromFolderDouble
from models.modeldoubleheight import STBVMM


def main(args):
    # Device choice (auto)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    print(f'Using device: {device}')

    # Create model
    model = STBVMM(img_size=384, patch_size=1, in_chans=3,
                   embed_dim=192, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                   window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                   drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                   norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                   use_checkpoint=False, img_range=1., resi_connection='1conv',
                   manipulator_num_resblk=1).to(device)

    # Load checkpoint
    if os.path.isfile(args.load_ckpt):
        print("=> loading checkpoint '{}'".format(args.load_ckpt))
        checkpoint = torch.load(args.load_ckpt)
        args.start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.load_ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.load_ckpt))
        assert (False)

    # Check saving directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)

    # Data loader
    dataset_mag = ImageFromFolderDouble(
        args.image_path, mag=args.mag, mode=args.mode, num_data=args.num_data, transform=None)
    data_loader = data.DataLoader(dataset_mag,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.workers,
                                  pin_memory=False)

    # Generate frames
    model.eval()

    for i, (xa, xb, mag_factor) in enumerate(data_loader):
        xa = xa.to(device)
        xb = xb.to(device)

        # forward
        with torch.no_grad():
            out_a, out_b = model(xa, xb)

        out_a = out_a.cpu().numpy()
        out_b = out_b.cpu().numpy()

        # save output images
        for j in range(out_a.shape[0]):
            output_a = Image.fromarray(np.uint8(out_a[j] * 255.0))
            output_a.save(os.path.join(save_dir, f'output_a_{i * args.batch_size + j}.png'))

            output_b = Image.fromarray(np.uint8(out_b[j] * 255.0))
            output_b.save(os.path.join(save_dir, f'output_b_{i * args.batch_size + j}.png'))

    print("Finished processing all images.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', default='ckpt/ckpt_e09.pth.tar', type=str)
    parser.add_argument('--save_dir', default='output/', type=str)
    parser.add_argument('--image_path', default='input/', type=str)
    parser.add_argument('--mag', default=20, type=int)
    parser.add_argument('--mode', default='static', type=str)
    parser.add_argument('--num_data', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--device', default='auto', type=str)
    args = parser.parse_args()

    main(args)
