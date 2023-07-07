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
import segmentation_models_pytorch as smp

class Chorus_dataset(Dataset):
    def __init__(self, path, split = "train", init = True, transform=True, use_sam = False):
        # Initialize the Dataset class
    
        self.transform_flag = transform
        self.path = path
        self.mask_dir = 'masks+/test' if split == 'test' else 'masks+/train' 
        self.img_size = (1024,1024) if use_sam else (640,1920) 
        self.images = []
        self.labels = []
        self.boxes = []
        if init == True:
            # Iterate over the files in the 'masks' directory
            for file in os.listdir(os.path.join(path, self.mask_dir)):
                fn = file.split('.npy')[0]

                # Append paths for images, labels, and boxes
                self.images.append(os.path.join(path, 'images', "{}.png".format(fn)))
                self.labels.append(os.path.join(path, self.mask_dir, "{}.npy".format(fn)))
                # self.boxes.append(os.path.join(self.path, 'boxes', file))
        elif init == False:
            for file in path:
                fn = os.path.split(file)[-1].split('png')[0]

                # Append paths for images, labels, and boxes
                self.images.append(os.path.join(file))
                self.labels.append(os.path.join('processed', 'oracle', "{}npy".format(fn)))
                # self.boxes.append(os.path.join(self.path, 'boxes', file))

            
            
        # Ensure the number of images is equal to the number of labels
        assert len(self.images) == len(self.labels)

    def __encode_image__(self, filepath):
        # Helper function to encode the image
        img = Image.open(filepath).convert("RGB")
        return img

    def __len__(self):
        # Return the length of the dataset
        return len(self.labels)

    def __transform__(self, image, mask):
        # Apply transformations to the image and mask

        # Resize the image and mask
        resize = T.Resize(size=self.img_size, interpolation=Image.NEAREST)
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
        image = self.__encode_image__(self.images[index])
        identifier = self.images[index]

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

        return {'image': img, 'mask': mask, 'id': identifier}

    
    

    
class Handler(Dataset):
    def __init__(self, X, Y, img_size=(640, 1920)):
        """
        Custom dataset handler for image and mask data.

        Args:
            X (list): List of image file paths.
            Y (list): List of mask file paths.
            img_size (tuple): Size to resize the image and mask.
        """
        self.X = X
        self.Y = Y
        self.img_size = img_size

    def __transform__(self, image, mask):
        """
        Apply transformations to the image and mask.

        Args:
            image (PIL.Image): Image object.
            mask (ndarray): Mask data.

        Returns:
            tuple: Transformed image and mask.
        """
        # Resize the image and mask
        resize = T.Resize(size=self.img_size, interpolation=Image.NEAREST)
        image = resize(image)
        mask = resize(mask)

        # Transform the image to a tensor
        image = TF.to_tensor(image)

        # Transform the mask to a tensor
        # mask = TF.to_tensor(mask)

        return image, mask

    def __getitem__(self, index):
        """
        Get the image, mask, and index at the given index.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Image, mask, and index.
        """
        # Retrieve the label and image at the given index
        label = torch.Tensor(np.load(self.Y[index]))
        image = Image.open(self.X[index]).convert("RGB")
        img, mask = self.__transform__(image, label)
        mask = mask.clamp(max=1.0)

        return img, mask, index

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.X)

    
class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        """
        Data management class for handling labeled and unlabeled data.

        Args:
            X_train (list): List of training image file paths.
            Y_train (list): List of training mask file paths.
            X_test (list): List of test image file paths.
            Y_test (list): List of test mask file paths.
            handler (Handler): Instance of the Handler class for data transformation.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num):
        """
        Initialize the labeled pool by randomly selecting a given number of samples.

        Args:
            num (int): Number of samples to initialize as labeled.

        Returns:
            None
        """
        # Generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_labeled_data(self):
        """
        Get the labeled data and their corresponding indices.

        Returns:
            tuple: Labeled indices and transformed labeled data.
        """
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        labeled_X = [self.X_train[idx] for idx in labeled_idxs]
        labeled_Y = [self.Y_train[idx] for idx in labeled_idxs]
        return labeled_idxs, self.handler(labeled_X, labeled_Y)

    def get_unlabeled_data(self):
        """
        Get the unlabeled data and their corresponding indices.

        Returns:
            tuple: Unlabeled indices and transformed unlabeled data.
        """
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        unlabeled_X = [self.X_train[idx] for idx in unlabeled_idxs]
        unlabeled_Y = [self.Y_train[idx] for idx in unlabeled_idxs]
        return unlabeled_idxs, self.handler(unlabeled_X, unlabeled_Y)

    def get_train_data(self):
        """
        Get the training data (labeled + unlabeled) and their corresponding indices.

        Returns:
            tuple: Labeled indices and transformed training data.
        """
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)

    def get_test_data(self):
        """
        Get the test data and their corresponding indices.

        Returns:
            tuple: Transformed test data.
        """
        return self.handler(self.X_test, self.Y_test)

    def cal_test_metrics(self, logits, mask):
        """
        Calculate evaluation metrics for the test data.

        Args:
            logits (torch.Tensor): Logits output of the model.
            mask (torch.Tensor): Ground truth mask.

        Returns:
            tuple: Intersection over Union (IoU) and F1-score.
        """
        prob_mask = logits.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # Compute true positive, false positive, false negative, and true negative 'pixels' for each class
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        # Calculate IoU and F1-score
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        return iou, f1


