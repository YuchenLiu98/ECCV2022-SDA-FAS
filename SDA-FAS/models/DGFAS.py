import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import sys
import numpy as np
from torch.autograd import Variable
import random
import os

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Feature_Embedder_patch(nn.Module):
    def __init__(self):
        super(Feature_Embedder_patch, self).__init__()
        self.layer4_patch = nn.Sequential(
            nn.Conv2d(384,384,kernel_size=4,stride=4,padding=0),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        #self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(384, 384)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input, norm_flag):
        feature = self.layer4_patch(input)
        feature = feature.reshape(-1,384)
        feature = self.bottleneck_layer(feature)
        if (norm_flag):
            # feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            feature = torch.div(feature, feature_norm)
        return feature

class Classifier_patch(nn.Module):
    def __init__(self):
        super(Classifier_patch, self).__init__()
        self.classifier_layer = nn.Linear(384, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out

class Feature_Embedder(nn.Module):
    def __init__(self):
        super(Feature_Embedder, self).__init__()
        self.layer4 = nn.Sequential(
            nn.Conv2d(384,512,3),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input, norm_flag):
        feature = self.layer4(input)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)
        if (norm_flag):
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            feature = torch.div(feature, feature_norm)
        return feature

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out

class Classifier_target(nn.Module):
    def __init__(self):
        super(Classifier_target, self).__init__()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input):
        self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
        classifier_out = self.classifier_layer(input)
        return classifier_out

class embedder_model(nn.Module):
    def __init__(self):
        super(embedder_model, self).__init__()
        
        self.embedder = Feature_Embedder()
        self.embedder_patch = Feature_Embedder_patch()
        self.classifier = Classifier()
        self.classifier_patch = Classifier_patch()

    def forward(self, input, norm_flag):
        feature = self.embedder(input, norm_flag)
        classifier_out = self.classifier(feature, norm_flag)
        return classifier_out, feature

if __name__ == '__main__': 
    x = Variable(torch.ones(1, 3, 256, 256))
    model = embedder_model()
    y, v = model(x, True)






