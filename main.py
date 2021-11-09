from core import cnn_network
from core import data_loader as dataloader

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def train(is_new_model, network):

    if(is_new_model == True):
        network.set_model("CNNmodel.ckpt")

    data = dataloader.DataLoader(".\\data\\train", ".\\data\\test")
    network.training_network(data.get_train_loader(), data.get_test_loader())

    plt.plot(network.get_train_losses, label = "train_losses")
    plt.plot(network.get_test_losses, label = "test_losses")
    plt.legend()

    network.save_model("CNNmodel.ckpt")


def main():

    input_train_str = input("Training a new model? ")

    try:
        input_train = bool(input_train_str)
    except ValueError:
        print("error")

    cnn_model = cnn_network.ConvolutionalNetwork()
    cnn_model.cuda(is_cuda = True)

    train(input_train, cnn_model)


if __name__ == '__main__':
    main()
