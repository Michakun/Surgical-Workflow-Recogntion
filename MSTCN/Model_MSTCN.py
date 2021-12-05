# Import useful libraries
from Data_MSTCN import *
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import copy
import torch.nn as nn
import torch.nn.functional as F

# Number of classes for each granularity
n_classes = [3,13,7,7]

class MS_TCN2(nn.Module):
    "Model from the MSTCN++ paper"
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, causal_conv):
        """
        Args:
            num_layers_PG (int): Number of layers in the first prediction network.
            num_layers_R (int): Number of layers in the refinement networks.
            num_R (int): Number of refinement networks.
            num_f_maps (int): Number of feature maps in the 1D conv layers.
            dim (int): Input dimension.
            num_classes (list): Number of classes for each granularity level.
            causal_conv (bool): Defines if the model uses causal or acausal convolutions.
        """
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes,causal_conv)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, sum(num_classes), 
                                                          num_classes,causal_conv)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

class Prediction_Generation(nn.Module):
    "First prediction network from the MSTCN++ architecture"
    def __init__(self, num_layers, num_f_maps, dim, num_classes,causal_conv=True):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)
        self.causal_conv = causal_conv
        if self.causal_conv :
            self.conv_dilated_1 = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**(num_layers-1-i), num_f_maps, num_f_maps,causal_conv)) for i in range(num_layers)])
            self.conv_dilated_2 = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps,causal_conv)) for i in range(num_layers)]) 
        else : 
            self.conv_dilated_1 = nn.ModuleList((nn.Conv1d(num_f_maps, num_f_maps, 3, 
                                                           padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
                                                           for i in range(num_layers)))
            self.conv_dilated_2 = nn.ModuleList((nn.Conv1d(num_f_maps, num_f_maps, 3, 
                                                           padding=2**i, dilation=2**i)
                                                           for i in range(num_layers)))
        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)))
        self.dropout = nn.Dropout()
        self.conv_out_phase = nn.Conv1d(num_f_maps, num_classes[0], 1)
        self.conv_out_step = nn.Conv1d(num_f_maps, num_classes[1], 1)
        self.conv_out_LV = nn.Conv1d(num_f_maps, num_classes[2], 1)
        self.conv_out_RV = nn.Conv1d(num_f_maps, num_classes[3], 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)
        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
        out_phase = self.conv_out_phase(f)
        out_step = self.conv_out_step(f)
        out_LV = self.conv_out_LV(f)
        out_RV = self.conv_out_RV(f)
        out = torch.cat((out_phase,out_step,out_LV,out_RV),axis=1)
        return out

class Refinement(nn.Module):
    "Refinement networks from the MSTCN++ architecture"
    def __init__(self, num_layers, num_f_maps, dim, num_classes,causal_conv=True):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps,causal_conv)) for i in range(num_layers)])
        self.conv_out_phase = nn.Conv1d(num_f_maps, num_classes[0], 1)
        self.conv_out_step = nn.Conv1d(num_f_maps, num_classes[1], 1)
        self.conv_out_LV = nn.Conv1d(num_f_maps, num_classes[2], 1)
        self.conv_out_RV = nn.Conv1d(num_f_maps, num_classes[3], 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out_phase = self.conv_out_phase(out)
        out_step = self.conv_out_step(out)
        out_LV = self.conv_out_LV(out)
        out_RV = self.conv_out_RV(out)
        out = torch.cat((out_phase,out_step,out_LV,out_RV),axis=1)
        return out
    
class MS_TCN(nn.Module):
    "Model from the MSTCN paper"
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes,causal_conv=True):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes,causal_conv)
        self.stages = nn.ModuleList([copy.deepcopy(SS_TCN(num_layers, num_f_maps, sum(num_classes), num_classes,causal_conv)) 
                                     for s in range(num_stages-1)])

    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    "Single stage TCN (first prediction network from the MSTCN architecture)"
    def __init__(self, num_layers, num_f_maps, dim, num_classes,causal_conv=True):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps,causal_conv)) for i in range(num_layers)])
        #self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.conv_out_phase = nn.Conv1d(num_f_maps, num_classes[0], 1)
        self.conv_out_step = nn.Conv1d(num_f_maps, num_classes[1], 1)
        self.conv_out_LV = nn.Conv1d(num_f_maps, num_classes[2], 1)
        self.conv_out_RV = nn.Conv1d(num_f_maps, num_classes[3], 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out_phase = self.conv_out_phase(out)
        out_step = self.conv_out_step(out)
        out_LV = self.conv_out_LV(out)
        out_RV = self.conv_out_RV(out)
        out = torch.cat((out_phase,out_step,out_LV,out_RV),axis=1)
        return out


class DilatedResidualLayer(nn.Module):
    "Dilated residual layers used in the MSTCN and MSTCN++ architectures"
    def __init__(self, dilation, in_channels, out_channels,causal_conv=True):
        super(DilatedResidualLayer, self).__init__()
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.causal_conv = causal_conv
        self.dropout = nn.Dropout()
        self.dilation = dilation
        if self.causal_conv :
            self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=(dilation*2), dilation=dilation)
        else :
            self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv :
            out = out[:,:,:-(self.dilation*2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out           
