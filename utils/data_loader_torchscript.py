from torchvision.datasets.folder import *
import numpy as np
import torch
from torchvision.transforms import Resize

from PIL import Image
import os

class ImageFromFolder(ImageFolder):
    def __init__(self, root, num_data=100000, preprocessing=False, transform=None, target_transform=None,
                 loader=default_loader):
        mag = np.loadtxt(os.path.join(root, 'train_mf.txt'))

        imgs = [(os.path.join(root, 'amplified', '%06d.png' % (i)),
                 os.path.join(root, 'frameA', '%06d.png' % (i)),
                 os.path.join(root, 'frameB', '%06d.png' % (i)),
                 os.path.join(root, 'frameC', '%06d.png' % (i)),
                 mag[i]) for i in range(num_data)]

        self.root = root
        self.imgs = imgs
        self.samples = self.imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.preproc = preprocessing

    def __getitem__(self, index):
        pathAmp, pathA, pathB, pathC, target = self.samples[index]
        sampleAmp, sampleA, sampleB, sampleC = self.loader(pathAmp), self.loader(pathA), self.loader(
            pathB), self.loader(pathC)

        # normalize
        sampleAmp = sampleAmp / 127.5 - 1.0
        sampleA = sampleA / 127.5 - 1.0
        sampleB = sampleB / 127.5 - 1.0
        sampleC = sampleC / 127.5 - 1.0

        # preprocessing
        if self.preproc:
            sampleAmp = preproc_poisson_noise(sampleAmp)
            sampleA = preproc_poisson_noise(sampleA)
            sampleB = preproc_poisson_noise(sampleB)
            sampleC = preproc_poisson_noise(sampleC)

        if self.transform is not None:
            sampleAmp = self.transform(Image.fromarray(sampleAmp))
            sampleA = self.transform(Image.fromarray(sampleA))
            sampleB = self.transform(Image.fromarray(sampleB))
            sampleC = self.transform(Image.fromarray(sampleC))

        sampleAmp, sampleA, sampleB, sampleC = torch.from_numpy(np.array(sampleAmp)), torch.from_numpy(
            np.array(sampleA)), torch.from_numpy(np.array(sampleB)), torch.from_numpy(np.array(sampleC))
        sampleAmp = sampleAmp.float()
        sampleA = sampleA.float()
        sampleB = sampleB.float()
        sampleC = sampleC.float()

        target = torch.from_numpy(np.array(target)).float()

        return sampleAmp, sampleA, sampleB, sampleC, target


def preproc_poisson_noise(image):
    nn = np.random.uniform(0, 0.3)  # 0.3
    n = np.random.normal(0.0, 1.0, image.shape)
    n_str = np.sqrt(image + 1.0) / np.sqrt(127.5)
    return image + nn * n * n_str


class ImageFromFolderTest(ImageFolder):
    def __init__(self, root, mag=10.0, mode='static', num_data=300, preprocessing=False, transform=None, target_transform=None, loader=default_loader):
        if mode=='static':
            imgs = [(root+'_%06d.png'%(1),
                     root+'_%06d.png'%(i+2),
                     mag) for i in range(num_data)]
        elif mode=='dynamic':
            imgs = [(root+'_%06d.png'%(i+1),
                     root+'_%06d.png'%(i+2),
                     mag) for i in range(num_data)]
        else:
            raise ValueError("Unsupported modes %s"%(mode))

        self.root = root
        self.imgs = imgs
        self.samples = self.imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.preproc = preprocessing
        self.desired_size = (364, 364)  # define the desired size

    def __getitem__(self, index):
        pathA, pathB, target = self.samples[index]
        sampleA, sampleB = self.loader(pathA), self.loader(pathB)

        # resize the images
        resize_transform = Resize(self.desired_size)
        sampleA = resize_transform(sampleA)
        sampleB = resize_transform(sampleB)

        # normalize
        sampleA = sampleA/127.5 - 1.0
        sampleB = sampleB/127.5 - 1.0

        # preprocessing
        if self.preproc:
            sampleA = preproc_poisson_noise(sampleA)
            sampleB = preproc_poisson_noise(sampleB)

        # to torch tensor
        tensor_transform = ToTensor()
        sampleA = tensor_transform(sampleA)
        sampleB = tensor_transform(sampleB)

        target = torch.tensor(target).float()

        return sampleA, sampleB, target

# Test
if __name__ == '__main__':
    dataset = ImageFromFolderTest('./../data/train', num_data=100, preprocessing=True)
    imageA, imageB, mag = dataset.__getitem__(0)

    import matplotlib.pyplot as plt
    plt.imshow(imageA.permute(1, 2, 0).numpy())
    plt.show()