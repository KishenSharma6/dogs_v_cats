import torch
import PIL #used to read images

import numpy as np
import os


class ImageDataset:
    def __init__(self, image_paths, targets, augmentations = None) -> None:
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations #torch vision transforms will suffice

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        target = self.targets[idx]
        image = PIL.Image.open(self.image_paths + str(idx) + '.jpg')

        #check for augmentations, apply changes to image if true
        if self.augmentations is not None:
            augmented = self.augmentations(image)
            image = augmented
        
        #convert image tensors to be in channel first format
        image = np.transpose(image, (2,0,1)).astype(np.float64)

        return {
            "image": torch.tensor(image),
            "target": torch.tensor(target)
        }


def standardize_image_name(path):
    """
    convert image names to numeric names
    """
    cat_idx = 0
    dog_idx = 12_500
    for img in os.listdir(path):
        if img.startswith('cat.'):    
            os.rename(os.path.join(path, img), os.path.join(path, str(cat_idx) + ".jpg"))
            cat_idx +=1
        if img.startswith('dog.'):    
            os.rename(os.path.join(path, img), os.path.join(path, str(dog_idx) + ".jpg"))
            dog_idx +=1