# Import useful libraries
from Data_PETRAW import *
import torchvision.models as models
import torch.nn as nn

# Create Identity function to replace the last layer of encoder models               
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

identity = Identity()

# Implementation of a single task classification model
class WRNet(nn.Module):
    """Video or Segmentation-based single-task model for the PETRAW Challenge"""
    
    def __init__(self, 
                 encoder: str = 'Resnet',
                 task : str = 'multi',
                 input_type : str = 'Images',
                 pre_trained : bool = True,
                 freeze : bool = True,
                 identity = identity):
        
        """
        Args:
            encoder (str): Name of the encoder backbone used for the classification network.
            task (str): Name of the wanted task.
            input_type (str): Indicates input type (Images or Segmentation masks).
            pretrained (bool): Indicates if the encoder must be pretrained on ImageNet.
            freeze (bool): Indicates if some encoder layers must be frozen. 
                           Modify the number of frozen layers if needed.
            identity (func) : Layer that replaces the last layer of the encoder backbone.
        """
        super(WRNet, self).__init__()
        
        # Useful variables
        self.task = task
        self.pre_trained = pre_trained
        self.freeze = freeze
        
        # Create backbone
        if encoder == 'Resnet':
            pretrained_model = models.resnet50(pretrained = self.pre_trained)
            # Modify first layer if input is Segmentation (1 channel instead of 3)
            if input_type != 'Images':
                pretrained_model.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
            # Modify last layer
            pretrained_model.fc = identity
            # Freeze layers
            if self.freeze == True :
                ct = 0
                for child in pretrained_model.children():
                    if ct < 7:
                        for param in child.parameters():
                            param.requires_grad = False
                    ct += 1
        elif encoder == 'ResneXt':
            pretrained_model = models.resnext50_32x4d(pretrained = self.pre_trained)
            # Modify first layer if input is Segmentation (1 channel instead of 3)
            if input_type != 'Images':
                pretrained_model.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
            # Modify last layer
            pretrained_model.fc = identity
            # Freeze layers
            if self.freeze == True :
                ct = 0
                for child in pretrained_model.children():
                    if ct < 7:
                        for param in child.parameters():
                            param.requires_grad = False
                    ct+=1
        elif encoder == 'VGG':
            pretrained_model = models.vgg19_bn(pretrained = self.pre_trained)
            # Modify first layer if input is Segmentation (1 channel instead of 3)
            if input_type != 'Images':
                pretrained_model.features[0] = nn.Conv2d(1,64,3,1,1)
            # Modify last layer
            pretrained_model.classifier._modules['6'] = identity
            if self.freeze == True :
                ct = 0
                for child in pretrained_model.children():
                    if ct < 38:
                        for param in child.parameters():
                            param.requires_grad = False
                    ct += 1
        
        # Define encoder backbone
        self.pretrained = pretrained_model
        
        # Define classification layers for single task WR model
        self.phase_layer = nn.Linear(2048,3)
        self.step_layer = nn.Linear(2048,13)
        self.LV_layer = nn.Linear(2048,7)
        self.RV_layer = nn.Linear(2048,7)
       
    def forward(self, x):
        output = self.pretrained(x)
        if self.task == 'phase' : 
            result = self.phase_layer(output)
        if self.task == 'step' : 
            result = self.step_layer(output)
        if self.task == 'LV' : 
            result = self.LV_layer(output)
        if self.task == 'RV' : 
            result = self.RV_layer(output)
        return result

# Implementation of the multitask classification model
class WRNet_Multi(nn.Module):
    """Video or Segmentation-based multi-task model for the PETRAW Challenge"""
    
    def __init__(self, 
                 encoder: str = 'Resnet',
                 hierarchical : bool = False,
                 input_type : str = 'Images',
                 pre_trained : bool = True,
                 freeze : bool = True,
                 identity=identity):
        
        """
        Args:
            encoder (str): Name of the encoder backbone used for the classification network.
            hierarchical (bool): Indicates whether to use a hierarchical model or not.
            input_type (bool): Indicates input type (Images or Segmentation masks).
            pretrained (bool): Indicates if the encoder must be pretrained on ImageNet.
            freeze (bool): Indicates if some encoder layers must be frozen. 
                           Modify the number of frozen layers if needed.
            identity (func) : Layer that replaces the last layer of the encoder backbone.
        """
        super(WRNet_Multi, self).__init__()
        
        # Useful variables
        self.pre_trained = pre_trained
        self.input_type = input_type
        if self.input_type !='Images':
            self.pre_trained = False
        self.freeze = freeze
        self.hierarchical = hierarchical
        
        # Create backbone
        if encoder == 'Resnet':
            pretrained_model = models.resnet50(pretrained = self.pre_trained)
            # Modify first layer if input is Segmentation (1 channel instead of 3)
            if input_type != 'Images':
                pretrained_model.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
            # Modify last layer
            pretrained_model.fc = identity
            # Freeze layers
            if self.freeze == True :
                ct = 0
                for child in pretrained_model.children():
                    if ct < 7:
                        for param in child.parameters():
                            param.requires_grad = False
                    ct += 1
        elif encoder == 'ResneXt':
            pretrained_model = models.resnext50_32x4d(pretrained = self.pre_trained)
            # Modify first layer if input is Segmentation (1 channel instead of 3)
            if input_type != 'Images':
                pretrained_model.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
            # Modify last layer
            pretrained_model.fc = identity
            # Freeze layers
            if self.freeze == True :
                ct = 0
                for child in pretrained_model.children():
                    if ct < 7:
                        for param in child.parameters():
                            param.requires_grad = False
                    ct+=1
        elif encoder == 'VGG':
            pretrained_model = models.vgg19_bn(pretrained = self.pre_trained)
            # Modify first layer if input is Segmentation (1 channel instead of 3)
            if input_type != 'Images':
                pretrained_model.features[0] = nn.Conv2d(1,64,3,1,1)
            # Modify last layer
            pretrained_model.classifier._modules['6'] = identity
            if self.freeze == True :
                ct = 0
                for child in pretrained_model.children():
                    if ct < 38:
                        for param in child.parameters():
                            param.requires_grad = False
                    ct += 1
        
        # Define multitask WR model
        self.pretrained = pretrained_model
        self.phase_layer = nn.Linear(2048,3)
        # Non-hierarchical model
        if self.hierarchical == False : 
            self.step_layer = nn.Linear(2048,13)
            self.LV_layer = nn.Linear(2048,7)
            self.RV_layer = nn.Linear(2048,7)
        # Hierarchical model
        elif self.hierarchical == True :
            self.step_layer = nn.Linear(2051,13)
            self.LV_layer = nn.Linear(2064,7)
            self.RV_layer = nn.Linear(2071,7)
            self.LV2_layer = nn.Linear(2078,7)
       
    def forward(self, x):
        # Non-hierarchical model
        if self.hierarchical == False : 
            output = self.pretrained(x)
            phase = self.phase_layer(output)
            step = self.step_layer(output)
            LV = self.LV_layer(output)
            RV = self.RV_layer(output)
        # Hierarchical model
        elif self.hierarchical == True :
            output = self.pretrained(x)
            phase = self.phase_layer(output)
            # use phase prediction for step prediction
            phase_feats = torch.cat([output, phase], dim=1)
            step = self.step_layer(phase_feats)
            # use step prediction for LV prediction
            step_feats = torch.cat([phase_feats, step], dim=1)
            LV = self.LV_layer(step_feats)
            # use LV prediction for RV prediction
            LV_feats = torch.cat([step_feats, LV], dim=1)
            RV = self.RV_layer(LV_feats)
            # use RV prediction for second LV prediction
            RV_feats = torch.cat([LV_feats, RV], dim=1)
            LV = self.LV2_layer(RV_feats)
        return phase,step,LV,RV
