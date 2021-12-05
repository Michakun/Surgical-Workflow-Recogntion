# Import useful libraries
import numpy as np
import pandas as pd
import random
import os

# Import torch libraries
import torch

# Useful dictionnaries
d0 = {'Idle': 0, 'Transfer Left to Right': 1, 'Transfer Right to Left': 2}
d1 = {'Block 1 L2R': 0, 'Block 1 R2L': 1,'Block 2 L2R': 2,'Block 2 R2L': 3,'Block 3 L2R': 4,'Block 3 R2L': 5,'Block 4 L2R': 6,'Block 4 R2L': 7,'Block 5 L2R': 8,'Block 5 R2L': 9,'Block 6 L2R': 10,'Block 6 R2L': 11,'Idle': 12}
d2 = {'Catch': 0, 'Drop': 1, 'Extract': 2, 'Hold': 3, 'Idle': 4, 'Insert': 5, 'Touch': 6}

class BatchGenerator(object):
    """Batch Generation using videos extracted frame features"""
    
    def __init__(self, path_source, gt_path, features_path, input_type, combination):
        """
        Args:
            path_source (str): Path to the directory that contains PETRAW Challenge training ddta.
            gt_path (str): Path to the directory that contains ground truth for procedural description.
            features_path (str) : Path to the directory that contains videos extracted features by another model (ResNet, etc.).
            input_type (str) : Type of the input.
            combination (str) : Type of feature combination.
        """
        # Initialize list of examples to be included in the batch
        self.list_of_examples = list()
        # Initialize index of the next loaded video
        self.index = 0
        # Define path to access features and GT
        self.input_type = input_type
        self.combination = combination
        self.gt_path = os.path.join(path_source,gt_path)
        if self.input_type == 'KV' or self.input_type == 'VS' : 
            self.features_path_0 = os.path.join(path_source,features_path[0])
            self.features_path_1 = os.path.join(path_source,features_path[1])
        elif self.input_type =='KVS' : 
            self.features_path_0 = os.path.join(path_source,features_path[0])
            self.features_path_1 = os.path.join(path_source,features_path[1])
            self.features_path_2 = os.path.join(path_source,features_path[2]) 
        else:
            self.features_path = os.path.join(path_source,features_path)
        
    # Function to reset the list of examples to be loaded within the batches
    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)
    
    # Function that checks if all videos have been loaded in the batches
    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False
    
    # Function that reads a txt file that contains the list of videos procedurial descriptions to be loaded in the batches 
    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)
    
    # Function that loads the batches 
    def next_batch(self, batch_size):
        # Define batch from list of examples and batch size
        batch = self.list_of_examples[self.index:self.index + batch_size]
        # Change the current index as a consequence
        self.index += batch_size
        
        # Initialize inputs and labels
        batch_input = []
        batch_target = []
        
        # For each video to be in the batch
        for vid in batch:
            # Define path to extracted video features & load data
            if self.input_type == 'VS' : 
                features_path_0 = self.features_path_0 + 'features' + vid.split('.')[0] + '.npz'
                features_path_1 = self.features_path_1 + 'features' + vid.split('.')[0] + '.npz'
                with open(features_path_0, "rb") as feature_file_0, open(features_path_1,"rb") as feature_file_1:
                    features_0 = np.load(feature_file_0)['arr_0'].T
                    features_1 = np.load(feature_file_1)['arr_0'].T
                    for i in range (features_0.shape[0]) :
                        features_0[i] = (features_0[i] - features_0[i].mean())/features_0[i].std()
                    for i in range (features_1.shape[0]) :
                        features_1[i] = (features_1[i] - features_1[i].mean())/features_1[i].std()
                    if self.combination == 'Add' :
                        features = features_0 + features_1
                    else :
                        features = np.concatenate((features_0,features_1),axis=0)
            
            elif self.input_type == 'KV' : 
                features_path_0 = self.features_path_0 + vid.split('.')[0] + '.kinematic'
                features_path_1 = self.features_path_1 + 'features' + vid.split('.')[0] + '.npz'
                with open(features_path_0, "rb") as feature_file_0, open(features_path_1,"rb") as feature_file_1:
                    features_0 = pd.read_table(feature_file_0)
                    features_0.drop('Frame',inplace=True,axis=1)
                    features_0 = np.asarray(features_0).T
                    features_1 = np.load(feature_file_1)['arr_0'].T
                    for i in range (features_0.shape[0]) :
                        features_0[i] = (features_0[i] - features_0[i].mean())/features_0[i].std()
                    for i in range (features_1.shape[0]) :
                        features_1[i] = (features_1[i] - features_1[i].mean())/features_1[i].std()
                    if self.combination == 'Add' :
                        features = features_0 + features_1
                    else :
                        features = np.concatenate((features_0,features_1),axis=0)
            
            elif self.input_type == 'KVS' : 
                features_path_0 = self.features_path_0 + vid.split('.')[0] + '.kinematic'
                features_path_1 = self.features_path_1 + 'features' + vid.split('.')[0] + '.npz'
                features_path_2 = self.features_path_2 + 'features' + vid.split('.')[0] + '.npz'
                with open(features_path_0, "rb") as feature_file_0, open(features_path_1,"rb") as feature_file_1, open(features_path_2,"rb") as feature_file_2:
                    features_0 = pd.read_table(feature_file_0)
                    features_0.drop('Frame',inplace=True,axis=1)
                    features_0 = np.asarray(features_0).T
                    features_1 = np.load(feature_file_1)['arr_0'].T
                    features_2 = np.load(feature_file_2)['arr_0'].T
                    for i in range (features_0.shape[0]) :
                        features_0[i] = (features_0[i] - features_0[i].mean())/features_0[i].std()
                    for i in range (features_1.shape[0]) :
                        features_1[i] = (features_1[i] - features_1[i].mean())/features_1[i].std()
                    for i in range (features_2.shape[0]) :
                        features_2[i] = (features_2[i] - features_2[i].mean())/features_2[i].std()
                    if self.combination == 'Add' :
                        features = features_1 + features_2
                        features = np.concatenate((features,features_0),axis=0)
                    else :
                        features = np.concatenate((features_1,features_2),axis=0)
                        features = np.concatenate((features,features_0),axis=0)
                        
            elif self.input_type == 'Kinematic' : 
                features_path = self.features_path + vid.split('.')[0] + '.kinematic'
                with open(features_path, "rb") as feature_file:
                    features = pd.read_table(feature_file)
                    features.drop('Frame',inplace=True,axis=1)
                    features = np.asarray(features).T
                    for i in range (features.shape[0]) :
                        features[i] = (features[i] - features[i].mean())/features[i].std()
        
            else:
                features_path_ = self.features_path + 'features' + vid.split('.')[0] + '.npz'
                with open(features_path_, "rb") as feature_file :
                    features = np.load(feature_file)['arr_0'].T
            
            # Define path to procedural description & load data
            gt_path_ = self.gt_path + vid
            with open(gt_path_,"rb") as label_file:
                label_vector = pd.read_table(label_file)
                labels = label_vector[label_vector['Frame']%1==0][['Phase','Step','Verb_Left','Verb_Right']]
                labels  = labels.replace({"Phase": d0})
                labels  = labels.replace({"Step": d1})
                labels  = labels.replace({"Verb_Left": d2})
                labels  = labels.replace({"Verb_Right": d2})
                labels = labels.to_numpy().T
                
                # Add input and label from the video
                batch_input.append(features)
                batch_target.append(labels)
        
        # Define length of each video
        length_of_sequences = [elem.shape[1] for elem in batch_target]
        
        # Create MSTCN input and label tensors given the video sizes (put label -100 if the video is shorter than the max length video)
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), 4,max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), 30,max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :,:np.shape(batch_target[i])[1]] = torch.from_numpy(batch_target[i])
            mask[i, :,:np.shape(batch_target[i])[0]] = torch.ones(30,np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask, length_of_sequences
