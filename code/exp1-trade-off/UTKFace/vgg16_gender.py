import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
from collections import OrderedDict

class VGG_16_2(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super(VGG_16_2, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('conv_1_1', nn.Conv2d(3, 64, 3, stride=1, padding=1) ),
            ('relu1', nn.ReLU()),
            ('conv_1_2', nn.Conv2d(64, 64, 3, stride=1, padding=1) ),
            ('relu2', nn.ReLU()),
            ('max1', nn.MaxPool2d(kernel_size=(2, 2))),
            ('conv_2_1',  nn.Conv2d(64, 128, 3, stride=1, padding=1) ),
            ('relu3', nn.ReLU()),
            ('conv_2_2',  nn.Conv2d(128, 128, 3, stride=1, padding=1) ),
            ('relu4', nn.ReLU()),
            ('max2', nn.MaxPool2d(kernel_size=(2, 2))), 
            ('conv_3_1', nn.Conv2d(128, 256, 3, stride=1, padding=1) ),
            ('relu5', nn.ReLU()),
            ('conv_3_2', nn.Conv2d(256, 256, 3, stride=1, padding=1) ),
            ('relu6', nn.ReLU()),
            ('conv_3_3', nn.Conv2d(256, 256, 3, stride=1, padding=1) ),
            ('relu7', nn.ReLU()),
            ('max3', nn.MaxPool2d(kernel_size=(2, 2))),
            ('conv_4_1',  nn.Conv2d(256, 512, 3, stride=1, padding=1) ),
            ('relu13', nn.ReLU()),
            ('conv_4_2',nn.Conv2d(512, 512, 3, stride=1, padding=1)),
            ('relu8', nn.ReLU()),
            ('conv_4_3', nn.Conv2d(512, 512, 3, stride=1, padding=1)),
            ('relu9', nn.ReLU()),
            ('max4', nn.MaxPool2d(kernel_size=(2, 2))),
            ('conv_5_1',  nn.Conv2d(512, 512, 3, stride=1, padding=1) ),
            ('relu10', nn.ReLU()),
            ('conv_5_2',nn.Conv2d(512, 512, 3, stride=1, padding=1)),
            ('relu11', nn.ReLU()),
            ('conv_5_3', nn.Conv2d(512, 512, 3, stride=1, padding=1)),
            ('relu12', nn.ReLU()),
            ('max5', nn.MaxPool2d(kernel_size=(2, 2)))

               
               
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(512, 256)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(256, 256)),
            ('relu7', nn.ReLU()),
            ('f8', nn.Linear(256,2))
            #('relu8', nn.ReLU()),
            
        ]))


        
        self.block_size = [2, 2, 3, 3, 3]
     
       


    def load_weights(self, path="pretrained/vgg_face_torch/VGG_FACE.t7"):
        """ Function to load luatorch pretrained

        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """

        

       
     
 
     
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = F.softmax(self.fc8(x), dim=1)
        return x


