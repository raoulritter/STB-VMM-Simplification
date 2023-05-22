import os
from PIL import Image
import torch.utils.data as data


class ImageFromFolderDouble(data.Dataset):
    def __init__(self, image_path, mode='static', num_data=1, preprocessing=False):
        super(ImageFromFolderDouble, self).__init__()
        self.image_path = image_path
        self.mode = mode
        self.num_data = num_data
        self.preprocessing = preprocessing
        self.image_files = self.load_image_files()

    def load_image_files(self):
        image_files = []

        if self.mode == 'static':
            image_files.append(os.path.join(self.image_path))
        elif self.mode == 'dynamic':
            for i in range(1, self.num_data + 1):
                image_files.append(os.path.join(self.image_path, 'frame_%06d.png' % i))

        return image_files

    def __getitem__(self, index):
        image_file = self.image_files[index]

        # Open image
        image = Image.open(image_file).convert("RGB")

        # Preprocessing
        if self.preprocessing:
            # Apply any necessary preprocessing steps here
            pass

        # Convert image to tensor
        image = self.to_tensor(image)

        # Stack the input image
        xa = image
        xb = image

        # Create magnitude factor tensor
        mag_factor = torch.tensor([1.0])

        return xa, xb, mag_factor

    def __len__(self):
        return len(self.image_files)

    def to_tensor(self, image):
        return torch.tensor(np.array(image).transpose(2, 0, 1))).float() / 255.0
