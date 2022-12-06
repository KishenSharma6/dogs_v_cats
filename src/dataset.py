import torch, torchvision
import torchvision.transforms.functional as fn

import PIL #used to read images

import numpy as np
import os,random


class ImageDataset:
    def __init__(self, image_paths, targets, augmentations = None) -> None:
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        # self.mean = self.calculate_mean()
        # self.std = self.calculate_std()

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        target = self.targets[idx]
        image = PIL.Image.open(self.image_paths + str(idx) + '.jpg')

        #convert PIL to tensor (channel first)
        image = torchvision.transforms.functional.pil_to_tensor(image)
        image = image.type(torch.float64)

        #check for augmentations, apply changes to image if true
        if self.augmentations is not None:
            augmented = self.augmentations(image)
            image = augmented        

        return {
            "image": torch.tensor(image),
            "target": torch.tensor(target)
        }

    def calculate_mean(self):
        mean = 0
        for i in range(len(self.targets)):
            mean += torch.mean(self.__getitem__(i)['image'], dim= [1,2])

        return mean/len(self.targets)
        
    def calculate_std(self):
        std = 0
        for i in range(len(self.targets)):
            std += torch.std(self.__getitem__(i)['image'], dim = [1,2])

        return std/len(self.targets)




def standardize_image_name(path, cat_start_idx= 0, dog_start_idx= 12_500):
    """
    convert image names to numeric names
    """
    cat_idx = cat_start_idx
    dog_idx = dog_start_idx
    for img in os.listdir(path):
        if img.startswith('cat.'):    
            os.rename(os.path.join(path, img), os.path.join(path, str(cat_idx) + ".jpg"))
            cat_idx +=1
        if img.startswith('dog.'):    
            os.rename(os.path.join(path, img), os.path.join(path, str(dog_idx) + ".jpg"))
            dog_idx +=1

def rename_jpg(path_to_read, path_to_write):
    files = os.listdir(path_to_read)

    for i, file in enumerate(files):
        os.rename(os.path.join(path_to_read, file), os.path.join(path_to_write, ''.join([str(i), '.jpg'])))
    print("rename complete")


def calculate_mean(training_data):
    mean = 0
    for i in range(len(training_data.targets)):
        mean += torch.mean(training_data.__getitem__(i)['image'], dim= [1,2])
    return mean/len(training_data.targets)

        
def calculate_std(training_data):
    std = 0
    for i in range(len(training_data.targets)):
        std += torch.std(training_data.__getitem__(i)['image'], dim = [1,2])
    return std/len(training_data.targets)
