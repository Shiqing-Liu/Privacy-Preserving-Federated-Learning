import torch
from torch import nn
import time, os

from models import *
from server import Server
from client import Client
from config import SERVER_HOST, SERVER_PORT, SAVE_PATH
from utils import LESS_DATA, SERVER_TEST_SIZE, SERVER_TRAIN_SIZE


def main():

    fed_config = {"C": 0.2,
                  "K": 3,
                  "R": 2,
                  "E": 2,
                  "B": 64,
                  "optimizer": torch.optim.Adam,
                  "criterion": nn.CrossEntropyLoss(),
                  "lr": 0.01,
                  "data_name": "CIFAR100",
                  "iid": True,
                  "shards_each": 2,
                  "ternary": True,
                  "personalized": True}

    if fed_config["ternary"]  & fed_config["data_name"] == "MNIST":
        model = Quantized_CNN(Net_3(), fed_config)
    elif fed_config["data_name"] == "MNIST":
        model = Net_2()
    elif fed_config["ternary"] & fed_config["data_name"] == "CIFAR100":
        model = Quantized_CNN(Net_4(100), fed_config)
    elif fed_config["data_name"] == "CIFAR100":
        model = Net_4(100)
    elif fed_config["ternary"] & fed_config["data_name"] == "CIFAR10":
        model = Quantized_CNN(Net_4(10), fed_config)
    elif fed_config["data_name"] == "CIFAR10":
        model = Net_4(10)
    else:
        raise AssertionError("No fitting model found. Check your parameters!")


    server = Server(model, fed_config, SERVER_HOST, SERVER_PORT)
    clients = []
    for i in range(fed_config["K"]):
        time.sleep(3)
        clients.append(Client(f"Client_{i + 1}", SERVER_HOST, SERVER_PORT))

    # Save configurations
    with open(os.path.join(SAVE_PATH, "configuration.txt"), 'w') as f:
        f.write(f"The following training was conducted:\n\n")
        for key, value in fed_config.items():
            f.write(f"{key}: {value}\n")
        f.write(f"model: {type(model)}\n")
        f.write(f"LESS_DATA: {LESS_DATA}\n")
        f.write(f"SERVER_TEST_SIZE: {SERVER_TEST_SIZE}\n")
        f.write(f"SERVER_TRAIN_SIZE: {SERVER_TRAIN_SIZE}\n")

    server.start()
    for client in clients: client.start()

    server.join()

    print("Finished!")


if __name__ == '__main__':
    main()
