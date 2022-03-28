from typing import Optional

import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class CifarLocalDataset(Dataset):
    def __init__(self, images, labels, num_classes, train=True):
        self.images = images
        self.labels = labels
        self.num_classes = num_classes
        self.train = train
        if train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index], mode='RGB')
        img = self.transform(img)
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)

    def get_subset_eq_distr(self, img_per_class: int):
        if img_per_class > len(self) // self.num_classes:
            raise ValueError("Not enough images")
        train_sorted_index = np.argsort(self.labels)
        train_img = self.images[train_sorted_index]
        train_label = self.labels[train_sorted_index]
        num_img_per_class = len(self) // self.num_classes
        subset_indexes = []
        for i in range(self.num_classes):
            subset_indexes += list(
                np.random.choice(num_img_per_class, img_per_class, replace=False) + num_img_per_class * i)

        subset_images = train_img[subset_indexes]
        subset_labels = train_label[subset_indexes]

        self.images = np.delete(train_img, subset_indexes, axis=0)
        self.labels = np.delete(train_label, subset_indexes)

        return CifarLocalDataset(subset_images, subset_labels, self.num_classes)

    def get_copy(self, train: Optional[bool] = None):
        is_train = train if train is not None else self.train
        return CifarLocalDataset(self.images, self.labels, self.num_classes, is_train)
