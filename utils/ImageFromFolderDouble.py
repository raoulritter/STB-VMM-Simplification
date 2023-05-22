import os
from PIL import Image
from torch.utils.data import Dataset

class ImageFromFolderDouble(Dataset):
    def __init__(self, root, mag, mode, num_data, transform=None):
        self.root = root
        self.mag = mag
        self.mode = mode
        self.num_data = num_data
        self.transform = transform
        self.image_list = self._load_image_list()

    def _load_image_list(self):
        image_list = []
        for i in range(self.num_data):
            image_a_path = os.path.join(self.root, f'frame{i:06d}.png')
            image_b_path = os.path.join(self.root, f'frame{i:06d}.png')

            if os.path.isfile(image_a_path) and os.path.isfile(image_b_path):
                image_list.append((image_a_path, image_b_path))

        return image_list

    def __getitem__(self, index):
        image_a_path, image_b_path = self.image_list[index]

        image_a = Image.open(image_a_path).convert('RGB')
        image_b = Image.open(image_b_path).convert('RGB')

        if self.transform is not None:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)

        if self.mode == 'static':
            return image_a, image_b, self.mag
        elif self.mode == 'dynamic':
            return image_a, image_b, self.mag[index]

    def __len__(self):
        return len(self.image_list)
