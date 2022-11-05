import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import seaborn as sn
import pandas as pd
from torchsummary import summary
import cv2
from PIL import Image

# import torchnet.meter.confusionmeter as cm

# Data augmentation and normalization for training
# Just normalization for validation & test
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'images_1028'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# lists for graph generation
epoch_counter_train = []
epoch_counter_val = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []






path = "whole_model_1028.pt"

model_ft = torch.load(path)
model_ft.eval()
# print(model_ft)
print("list:",list(model_ft.children())[-1])

# Test the accuracy with test data
correct = 0
total = 0

# test= Image.open('test2.jpg')
# print("test_PIL:",np.array(test))
test_cv = cv2.imread('test6.jpg')[:,:,::-1]
test = Image.fromarray(test_cv)
pre_process = transforms.Compose([transforms.Resize(224),
        
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

print(class_names)
test = pre_process(test)
test = test.unsqueeze(0)
test = test.to(device)
print(test.shape)
outputs_test = model_ft(test)
print('outputs_test.data:',outputs_test.data)
_, pre_test = torch.max(outputs_test.data, 1)
class_index = int(pre_test.cpu().numpy())
print(class_names[class_index])

# pre_test = transforms.Resize(pre_test)
# pre_test = transforms.ToTensor(pre_test)
# with torch.no_grad():
#     for i, (inputs, labels) in enumerate(dataloaders['test']):
#         inputs = inputs.to(device)
#         print('inputs:',inputs.shape)
#         labels = labels.to(device)
#         outputs = model_ft(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         # print('pred:',predicted)
#         # print('labels:',labels)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the test images: %d %%' % (
#         100 * correct / total))

# #Class wise testing accuracy
# class_correct = list(0. for i in range(16))
# class_total = list(0. for i in range(16))
# with torch.no_grad():
#     for i, (inputs, labels) in enumerate(dataloaders['test']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model_ft(inputs)
#             _, predicted = torch.max(outputs, 1)
#             point = (predicted == labels).squeeze()
#             for j in range(len(labels)):
#                 label = labels[j]
#                 class_correct[label] += point[j].item()
#                 class_total[label] += 1
#
# for i in range(16):
#     print('Accuracy of %5s : %2d %%' % (
#         class_names[i], 100 * class_correct[i] / class_total[i]))


# #Get the confusion matrix for testing data
# confusion_matrix = cm.ConfusionMeter(16)
# with torch.no_grad():
#     for i, (inputs, labels) in enumerate(dataloaders['test']):
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         outputs = model_ft(inputs)
#         _, predicted = torch.max(outputs, 1)
#         confusion_matrix.add(predicted, labels)
#     print(confusion_matrix.conf)
#
# #Confusion matrix as a heatmap
# con_m = confusion_matrix.conf
# df_con_m = pd.DataFrame(con_m, index= [i for i in class_names], columns = [i for i in class_names])
# sn.set(font_scale= 1.1)
# sn.heatmap(df_con_m, annot=True,fmt='g' ,  annot_kws={"size" : 10}, cbar = False, cmap="Blues")