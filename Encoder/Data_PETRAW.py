# Import useful libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Any, Callable, Optional

# Import torch libraries
import torch
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset

# Useful dictionnaries to convert labels into integers
d0 = {'Idle': 0, 'Transfer Left to Right': 1, 'Transfer Right to Left': 2}
d1 = {'Block 1 L2R': 0, 'Block 1 R2L': 1,'Block 2 L2R': 2,'Block 2 R2L': 3,'Block 3 L2R': 4,'Block 3 R2L': 5,'Block 4 L2R': 6,'Block 4 R2L': 7,'Block 5 L2R': 8,'Block 5 R2L': 9,'Block 6 L2R': 10,'Block 6 R2L': 11,'Idle': 12}
d2 = {'Catch': 0, 'Drop': 1, 'Extract': 2, 'Hold': 3, 'Idle': 4, 'Insert': 5, 'Touch': 6}


# Implementation of the PETRAW dataset class
class PETRAW_Dataset(VisionDataset) : 
    """Dataset of PETRAW Challenge"""
    
    def __init__(self,
                 root: str,
                 image_folder: str,
                 label_folder: str,
                 n_samples: list,
                 transforms: Optional[Callable] = None,
                 task: str = 'multi',
                 stride: int = 1,
                 input_type: str = 'Images'
                 ):
    
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
            input_type (str) : Data type of the input (Images or Segmentations)
        """
        
        super().__init__(root, transforms)
        
        # Useful variables
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.n_samples = n_samples
        self.task = task
        self.input_type = input_type
        
        # Define paths to images and labels folders
        image_folder_path = os.path.join(self.root,self.image_folder)
        label_folder_path = os.path.join(self.root,self.label_folder)

        # Define list of folders (i.e. list of videos) containing images 
        self.image_folders = sorted([elem for elem in os.listdir(image_folder_path)])
        
        # Create list of paths to all images that will be loaded into the train and validation datasets
        self.image_list = []
        
        # Define file extension according to the input type
        if self.input_type == 'Images':
            self.extension = 'jpg'
        else:
            self.extension = 'npz'    
        
        # Create list of paths to samples in the training/validation set 
        for i in (n_samples) :
            self.image_list += [os.path.join(image_folder_path,self.image_folders[i],elem) 
                                for elem in sorted(os.listdir(os.path.join(image_folder_path,self.image_folders[i]))) 
                                if elem[-3:]==self.extension and int(elem[9:][:-4])%stride==0]    
        
        # Create list of tuples associated to each image in the image list 
        # (path to the procedural description of the video containing the image, 
        # frame number of the image in the procedural description of the video)
        self.label_list = [(os.path.join(label_folder_path,elem[-21:-18]+'.txt'),int(elem[-8:-4])) 
                           for elem in self.image_list]
        
        # Avoid batches of size 1 when using a batch size of 2
        if len(self.image_list)%2 == 1 : 
            self.image_list = self.image_list[:-1]
            self.label_list = self.label_list[:-1]
                    
    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> Any:
        # Create image/label couples into samples using list of paths previously defined
        image_path = self.image_list[index]
        label_path = self.label_list[index][0]
        label_num = self.label_list[index][1]
        with open(image_path, "rb") as image_file, open(label_path,"rb") as label_file:
            # Load image
            if self.input_type == 'Images':
                image = Image.open(image_file)
            else:
                image = np.load(image_file)['arr_0'].squeeze()

            # Load corresponding label
            label_vector = pd.read_table(label_file)
            labels = label_vector[label_vector['Frame']==label_num][['Phase','Step','Verb_Left','Verb_Right']]
            labels  = labels.replace({"Phase": d0})
            labels  = labels.replace({"Step": d1})
            labels  = labels.replace({"Verb_Left": d2})
            labels  = labels.replace({"Verb_Right": d2})
            labels = torch.tensor(labels.to_numpy().reshape(-1))
            
            # Create sample
            if self.task == 'multi':
                sample = {"image": image, "labels": labels}
            elif self.task == 'phase':
                sample = {"image": image, "labels": labels[0]}
            elif self.task == 'step':
                sample = {"image": image, "labels": labels[1]}
            elif self.task == 'LV':
                sample = {"image": image, "labels": labels[2]}
            elif self.task == 'RV':
                sample = {"image": image, "labels": labels[3]}
            # apply transformations
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["image"] = sample["image"].float()
            return sample
