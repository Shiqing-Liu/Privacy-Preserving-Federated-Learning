import torch
import torchvision
from torchvision import transforms
from client import Client
from server import Server
from models import *
from torch import nn
import logging
import time
import os
from threading import Thread
from server import Server
from client import Client
from config import SERVER_HOST, SERVER_PORT


def main():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    fed_config = {"C": 0.6,
                  "K": 5,
                  "R": 15,
                  "E": 3,
                  "B": 64,
                  "optimizer": torch.optim.SGD,
                  "criterion": nn.CrossEntropyLoss(),
                  "lr": 0.01,
                  "data_name": "MNIST",
                  "iid": False,
                  "shards_each": 2,
                  "ternary" : True}
    if fed_config["ternary"]:
        model = Quantized_CNN(Net_3(), fed_config)
    else:
        model = Net_2

    server = Server(model, fed_config, SERVER_HOST, SERVER_PORT)
    clients = [Client(f"Client_{i + 1}", SERVER_HOST, SERVER_PORT) for i in range(fed_config["K"])]

    server.start()
    for client in clients: client.start()

    server.join()

    print("Finished!")


if __name__ == '__main__':
    main()
