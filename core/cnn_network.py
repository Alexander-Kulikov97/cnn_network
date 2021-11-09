import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.__train_losses = []
        self.__test_losses = []
        self.__train_correct = []
        self.__test_correct = []

        self.conv1=nn.Conv2d(3, 6, 3, 1)
        self.conv2=nn.Conv2d(6, 16, 3, 1)
        self.fc1=nn.Linear(16 * 54 * 54, 120) 
        self.fc2=nn.Linear(120, 84)
        self.fc3=nn.Linear(84, 20)
        self.fc4=nn.Linear(20, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        
        return F.log_softmax(X, dim=1)

    def training_network(self, train_loader, test_loader):
        epochs = 5
        start_time = time.time()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.001) 

        for i in range(epochs):
            trn_corr = 0
            tst_corr = 0
            for b, (X_train, y_train) in enumerate(train_loader):
                X_train, y_train = X_train.cuda(), y_train.cuda()

                b += 1  # test batch sizes.

                y_pred = self(X_train)
                loss = criterion(y_pred,y_train)
                #true predictions
                predicted = torch.max(y_pred.data,1)[1]
                batch_corr = (predicted==y_train).sum()
                trn_corr += batch_corr
                
                #update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #print interim results
                if b%200==0:
                    print(f"epoch: {i} loss: {loss.item} batch: {b} accuracy: {trn_corr.item()*100/(10*b):7.3f}%")

            loss = loss.cpu().detach().numpy()
            self.__train_losses.append(loss)
            self.__train_correct.append(trn_corr)
            
            #test data
            with torch.no_grad():
                for b, (X_test,y_test) in enumerate(test_loader):
                    X_test, y_test = X_test.cuda(), y_test.cuda()

                    y_val = self(X_test)
                    loss = criterion(y_val, y_test)
                    
                    predicted = torch.max(y_val.data, 1)[1]
                    btach_corr = (predicted==y_test).sum()
                    tst_corr += btach_corr
                    
                loss = loss.cpu().detach().numpy()
                self.__test_losses.append(loss)
                self.__test_correct.append(tst_corr)
                
        print(f'\nDuration: {time.time() - start_time:.0f} seconds') 

    def get_test_losses(self):
        return self.__test_losses

    def get_test_correct(self):
        return self.__test_correct

    def get_train_losses(self):
        return self.__train_losses

    def get_train_correct(self):
        return self.__train_correct

    def cuda(self, is_cuda):
        if (torch.cuda.is_available() and is_cuda):
            device = torch.device("cuda") 
        else:
            device = torch.device("cpu")

        self.to(device)

    def save_model(self, name):
        torch.save(self.state_dict(), '.\\{name}.ckpt')

    def set_model(self, path):
        simplenet_state_dict = torch.load(path)
        self.load_state_dict(simplenet_state_dict)