# Importing useful libraries
import numpy as np
from typing import Any, Callable, Optional
from PIL import Image
import os
import copy
import torch
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset


# Implementation of the PETRAW dataset class
class PETRAW_Dataset(VisionDataset) : 
    """Dataset of PETRAW Challenge"""
    
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 transforms: Optional[Callable] = None,
                 n_samples: int = 3,
                 stride: int = 5):
    
        
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            label_folder (str): Name of the folder that contains the labels in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
                                                       a sample and returns a transformed version. 
                                                       E.g, ``transforms.ToTensor`` for images. 
                                                       Defaults to None.
            n_samples (list): List of sequences' indexes to use to create the training or validation set.
            task (str) : Classification task that is wanted.
            stride (int) : Interval between frames loaded in the training or validation set.
        """
        
        super().__init__(root, transforms)
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.n_samples = n_samples
        
        # Define paths to images and masks folders
        image_folder_path = os.path.join(self.root,self.image_folder)
        mask_folder_path = os.path.join(self.root,self.mask_folder)
        
        # Define list of folders (i.e. list of videos) containing images and masks for segmentation
        self.image_folders = sorted([elem for elem in os.listdir(image_folder_path)])
        self.mask_folders = sorted([elem for elem in os.listdir(mask_folder_path)])
        
        # Create list of paths to all images and masks that will be loaded into the train and validation datasets
        self.image_list = []
        self.mask_list = []
        
        # Create list of paths to samples in the training set using the first two-thirds of the image/mask folders list
        for i in (n_samples) :
            self.image_list += [os.path.join(image_folder_path,self.image_folders[i],elem) 
                                for elem in sorted(os.listdir(os.path.join(image_folder_path,self.image_folders[i]))) 
                                if elem[-3:]=='jpg' and int(elem[9:][:-4])%stride==0]
            self.mask_list += [os.path.join(mask_folder_path,self.mask_folders[i],elem)
                                for elem in sorted(os.listdir(os.path.join(mask_folder_path,self.mask_folders[i]))) 
                                if elem[-3:]=='npz' and int(elem[9:][:-4])%stride==0]
        
        # Avoid batches of size 1 when using a batch size of 2
        if len(self.image_list)%2 == 1 : 
            self.image_list = self.image_list[:-1]
            self.mask_list = self.mask_list[:-1]        
    
    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> Any:
        # Create image/mask couples into samples using list of paths previously defined
        image_path = self.image_list[index]
        mask_path = self.mask_list[index]
        with open(image_path, "rb") as image_file, open(mask_path,"rb") as mask_file:
            # load image
            image = Image.open(image_file)
            image = np.asarray(image)
            image_ = copy.deepcopy(image)
            # load mask
            mask = np.load(mask_file)['arr_0']
            # create sample
            sample = {"image": image_, "mask": mask}
            # apply transformations
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])
                sample["mask"] = torch.argmax(sample["mask"],axis=0)
            return sample
