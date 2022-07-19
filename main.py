import torch
from torch import nn
import time

from models import *
from server import Server
from client import Client
from config import SERVER_HOST, SERVER_PORT


def main():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    ## Warum 5 Clints bei K = 3???

    # test run parameter
    # fed_config = {"C": 0.6,
    #               "K": 5,
    #               "R": 15,
    #               "E": 4,
    #               "B": 64,
    #               "optimizer": torch.optim.Adam,
    #               "criterion": nn.CrossEntropyLoss(),
    #               "lr": 0.01,
    #               "data_name": "MNIST",
    #               "iid": True,
    #               "shards_each": 2,
    #               "ternary": True}
    fed_config = {"C": 0.4,
                  "K": 3,
                  "R": 1,
                  "E": 2,
                  "B": 64,
                  "optimizer": torch.optim.Adam,
                  "criterion": nn.CrossEntropyLoss(),
                  "lr": 0.01,
                  "data_name": "MNIST",
                  "iid": False,
                  "shards_each": 2,
                  "ternary": True}
    if fed_config["ternary"]:
        model = Quantized_CNN(Net_3(), fed_config)
    else:
        model = Net_2

    server = Server(model, fed_config, SERVER_HOST, SERVER_PORT)
    clients = []
    for i in range(fed_config["K"]):
        time.sleep(3)
        clients.append(Client(f"Client_{i + 1}", SERVER_HOST, SERVER_PORT))

    server.start()
    for client in clients: client.start()

    server.join()

    print("Finished!")


if __name__ == '__main__':
    main()
