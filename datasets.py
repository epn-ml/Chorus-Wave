#!/usr/bin/env python

"""datasets.py : Contains pytorch dataset inherited classes for training pytorch NNs"""

__author__      = "Sahib Julka <sahib.julka@uni-passau.de>"
__copyright__   = "GPL"


from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import torch
import numpy as np # linear algebra



class Dataset(Dataset):
    def __init__(self, path, transform=True):
        # Initialize the Dataset class
        self.transform_flag = transform
        self.path = path
        self.images = []
        self.labels = []
        self.boxes = []

        # Iterate over the files in the 'masks' directory
        for file in os.listdir(os.path.join(self.path, 'masks')):
            fn = file.split('.npy')[0]

            # Append paths for images, labels, and boxes
            self.images.append(os.path.join(self.path, 'chorus', "{}.png".format(fn)))
            self.labels.append(os.path.join(self.path, 'masks', "{}.npy".format(fn)))
            # self.boxes.append(os.path.join(self.path, 'boxes', file))

        # Ensure the number of images is equal to the number of labels
        assert len(self.images) == len(self.labels)

    def __encode_image(self, filepath):
        # Helper function to encode the image
        img = Image.open(filepath).convert("RGB")
        return img

    def __len__(self):
        # Return the length of the dataset
        return len(self.labels)

    def __transform__(self, image, mask):
        # Apply transformations to the image and mask

        # Resize the image and mask
        resize = T.Resize(size=(1024, 1024), interpolation=Image.NEAREST)
        image = resize(image)
        mask = resize(mask)

        # Transform the image to a tensor
        image = TF.to_tensor(image)
        # Transform the mask to a tensor
        # mask = TF.to_tensor(mask)

        return image, mask

    def __getitem__(self, index):
        # Retrieve the label and image at the given index
        label = torch.Tensor(np.load(self.labels[index]))
        image = self.__encode_image(self.images[index])

        # boxes = torch.Tensor(np.load(self.boxes[index]))
        # Reshape boxes

        if self.transform_flag == True:
            # Apply transformations if transform_flag is True
            img, mask = self.__transform__(image, label)
        else:
            # Convert image to tensor without transformations
            img = TF.to_tensor(image)
            mask = label
        mask = mask.clamp(max=1.0)

        return {'image': img, 'mask': mask}
