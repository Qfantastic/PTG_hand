import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import time
import os
import copy
import seaborn as sn
import pandas as pd
#import torchnet.meter.confusionmeter as cm

# Data augmentation and normalization for training
# Just normalization for validation & test

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


data_transforms = {
    'raw': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ]),
    'train': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5484, 0.4518, 0.3903], [0.1798, 0.2289, 0.2124]),
        #transforms.ColorJitter(brightness=0.3,hue= 0.5),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([AddGaussianNoise(0., 1.)], p = 0.5)
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5484, 0.4518, 0.3903], [0.1798, 0.2289, 0.2124])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5484, 0.4518, 0.3903], [0.1798, 0.2289, 0.2124]),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),

        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'images_1104'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
print("class_names:",class_names)
print(len(class_names))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# image_datasets_train = datasets.ImageFolder(os.path.join(data_dir, 'train'),
#                                           data_transforms['raw'])
# dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=16,
#                                              shuffle=True, num_workers=4)
#
# mean, std = get_mean_and_std(dataloaders_train)
# print("mean:",mean)
# print("std:",std)


#lists for graph generation
epoch_counter_train = []
epoch_counter_val = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

#Train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch +1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            #For graph generation
            if phase == "train":
                train_loss.append(running_loss/dataset_sizes[phase])
                train_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_train.append(epoch)
            if phase == "val":
                val_loss.append(running_loss/ dataset_sizes[phase])
                val_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_val.append(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #for printing        
            if phase == "train":    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "val":    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_model = copy.deepcopy(model)
                print("Adding the best model......")
                print("saving model ......")
                torch.save(best_model, f"whole_model_1104_{epoch}.pt")
            scheduler.step()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    #print("saving model ......")
    torch.save(best_model,"whole_model_1103_3.pt")
    return model

#Using a model pre-trained on ImageNet and replacing it's final linear layer

#For resnet18
# model_ft = models.resnet18(weights='IMAGENET1K_V1')
model_ft = models.resnet18(pretrained=True)
# model_ft = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 28)

#for VGG16_BN
#model_ft = models.vgg16_bn(pretrained=True)
#model_ft.classifier[6].out_features = 16

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Using Adam as the parameter optimizer
optimizer_ft = optim.Adam(model_ft.parameters(), lr = 0.001, betas=(0.9, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)

# torch.save(model_ft.state_dict(), "best.pt")
#Plot the train & validation losses
# plt.figure(1)
# plt.title("Training Vs Validation Losses")
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.plot(epoch_counter_train,train_loss,color = 'r', label="Training Loss")
# plt.plot(epoch_counter_val,val_loss,color = 'g', label="Validation Loss")
# plt.legend()
# plt.show()

#Plot the accuracies in train & validation
# plt.figure(2)
# plt.title("Training Vs Validation Accuracies")
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.plot(epoch_counter_train,train_acc,color = 'r', label="Training Accuracy")
# plt.plot(epoch_counter_val,val_acc,color = 'g', label="Validation Accuracy")
# plt.legend()
# plt.show()

#Test the accuracy with test data
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

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

