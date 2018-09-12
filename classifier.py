import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torchvision.models as models

# REF: https://github.com/arnoweng/CheXNet/blob/master/model.py

class Classifier(nn.Module):        
    def __init__(self, out_size):
        super(Classifier, self).__init__()
        self.densenet = models.densenet121(pretrained=True)

        for name, child in self.densenet.features.named_children():  
            if name == 'denseblock3':
#                 print(name)
                break
            for param in child.parameters():
                param.requires_grad_(False)
#         print('===print parameters requires grad===')
#         # REF: https://medium.com/@14prakash/almost-any-image-classification-problem-using-pytorch-i-am-in-love-with-pytorch-26c7aa979ec4     
#         # To view which layers are freeze and which layers are not freezed:
#         for name, child in self.densenet.features.named_children():  
#             for name_2, params in child.named_parameters():
#                 print(name, name_2, params.requires_grad)
                
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size)#,
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet(x)
        return x