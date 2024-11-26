# https://analyticsindiamag.com/how-to-implement-convolutional-autoencoder-in-pytorch-with-cuda/
# https://stackoverflow.com/questions/59924310/load-custom-data-from-folder-in-dir-pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.utils import make_grid
import time
import copy




from torchvision import datasets, models, transforms


from PIL import Image
def load_image(image_file):
    img = Image.open(image_file)
    return img



# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False
# def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
#     # Initialize these variables which will be set in this if statement. Each of these
#     #   variables is model specific.
#     model_ft = None
#     input_size = 0

#     if model_name == "resnet":
#         """ Resnet18
#         """
#         model_ft = models.resnet18(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs, num_classes)
#         input_size = 224

#     elif model_name == "alexnet":
#         """ Alexnet
#         """
#         model_ft = models.alexnet(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.classifier[6].in_features
#         model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
#         input_size = 224

#     elif model_name == "vgg":
#         """ VGG11_bn
#         """
#         model_ft = models.vgg11_bn(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.classifier[6].in_features
#         model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
#         input_size = 224

#     elif model_name == "squeezenet":
#         """ Squeezenet
#         """
#         model_ft = models.squeezenet1_0(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
#         model_ft.num_classes = num_classes
#         input_size = 224

#     elif model_name == "densenet":
#         """ Densenet
#         """
#         model_ft = models.densenet121(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.classifier.in_features
#         model_ft.classifier = nn.Linear(num_ftrs, num_classes)
#         input_size = 224

#     elif model_name == "inception":
#         """ Inception v3
#         Be careful, expects (299,299) sized images and has auxiliary output
#         """
#         model_ft = models.inception_v3(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         # Handle the auxilary net
#         num_ftrs = model_ft.AuxLogits.fc.in_features
#         model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
#         # Handle the primary net
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs,num_classes)
#         input_size = 299

#     else:
#         print("Invalid model name, exiting...")
#         exit()

#     return model_ft, input_size



# read an image



label_map={
    0:"Chickenpox",
    1:"Measles",
    2:"Monkeypox",
    3:"Normal"
}
classes = ('Chickenpox', 'Measles', 'Monkeypox', 'Normal')
PATH = 'models/resnet18_net.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.Resize((64,64)),
                                     transforms.ToTensor()])





def load_model():
    '''

    load a model 
    by default it is resnet 18 for now
    '''

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.to(device)

    model.load_state_dict(torch.load(PATH,map_location=device))
    model.eval()
    return model





def image_loader(image_name):
    """load image, returns cuda tensor"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    picture = Image.open(image_name)
    image = data_transform(picture)
    images=image.reshape(1,1,64,64)
    new_images = images.repeat(1, 3, 1, 1)
    return new_images
    
def predict(model, image_name):
    '''

    pass the model and image url to the function
    Returns: a list of pox types with decreasing probability
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    picture = Image.open(image_name)
    image = data_transform(picture)
    images=image.reshape(1,1,64,64)
    new_images = images.repeat(1, 3, 1, 1)

    outputs=model(new_images)

    _, predicted = torch.max(outputs, 1)
    ranked_labels=torch.argsort(outputs,1)[0]
    probable_classes=[]
    for label in ranked_labels:
        probable_classes.append(classes[label.numpy()])
    probable_classes.reverse()
    return probable_classes



