#-------------Import required librarires------------------------
import numpy as np
import random
import torch
from torchvision import models, transforms, datasets
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fn
import os
from PIL import Image
from torchvision.models import resnet50
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

total_epochs = 3
batch = 100
learning_rate = 0.001


def used_device():
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device}')
    return device


def pretrained_raw():
    print("")
    print("Enter 0 to train the model with random weights or any other number for pretrained weights")
    pretrained = int(input())
    return pretrained


def used_model():
    #weights = ResNet50_Weights.DEFAULT
    if pretrained == 0:
        model = resnet50(weights=None)
    else:
        model = resnet50(pretrained=True)
        #model = resnet50(weights = weights)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(
        7, 7), stride=(2, 2), padding=(5, 5), bias=0.01)
    #model.conv1.weight.data.fill_(0.0000000000001)
    #model.fc = nn.Linear(2048, 10, bias=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=10, bias=True)
    model = model.to(device)
    return model


def used_data():
    print("")
    print("Dataset is being initialized")
    dataset_training = torchvision.datasets.MNIST(root='./data',
                                                  train=True,
                                                  transform=transforms.ToTensor(),
                                                  download=True)
    dataset_testing = datasets.MNIST(root='./data',
                                     train=False,
                                     transform=transforms.ToTensor())

    target_dataset = random.choices(dataset_training, k=50000)

    dataloader_training = torch.utils.data.DataLoader(dataset=target_dataset,
                                                      batch_size=batch,
                                                      shuffle=True)
    dataloader_testing = torch.utils.data.DataLoader(dataset=dataset_testing,
                                                     batch_size=batch,
                                                     shuffle=True)
    #print(dataset_training)
    return dataloader_training, dataloader_testing, target_dataset


def used_len():
    train_len = len(dataloader_training)
    test_len = len(dataloader_testing)
    return train_len, test_len


def loss_opt():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    return criterion, optimizer


def training():
    model.train()
    train_len = len(target_dataset)
    for epoch in range(total_epochs):
        correct = 0.0
        corr = 0.0
        for i, (inputs, labels) in enumerate(dataloader_training):
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            output_idx = torch.argmax(output, dim=1)
            corr = (labels == output_idx).sum()
            correct = (corr + correct).item()
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print("")
                print("Epoch =", (epoch+1), "of", (total_epochs))
                print("Step =", (i+1)*100, "of", (train_len))
                print("Loss =", loss.item())
                print("")
                #print("Accuracy =" (correct/train_len)*100,"%"]


def testing():
    with torch.no_grad():
        model.eval()
        total_correct = 0
        correct = 0
        samples = 0
        for inputs, labels in dataloader_testing:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs_idx = torch.argmax(outputs.data, dim=1)
            samples += labels.size(0)
            correct = (outputs_idx == labels).sum()
            total_correct = (correct+total_correct).item()
        accuracy = 100.0 * total_correct / samples
        print("Accuracy with 10000 images =", accuracy, "%")


device = used_device()
pretrained = pretrained_raw()
model = used_model()
dataloader_training, dataloader_testing, target_dataset = used_data()
train_len, test_len = used_len()
criterion, optimizer = loss_opt()
start_time = time.time()
training()
end_time = time.time()
training_time = end_time - start_time
testing()
print("Time for training =", training_time,"s")