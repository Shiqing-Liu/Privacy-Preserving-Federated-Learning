import torch
from collections import Counter
import pickle
import struct
from socket import socket, AF_INET, SOCK_STREAM
import matplotlib.pyplot as plt
import numpy as np

from config import SERVER_HOST, SERVER_PORT
from utils import get_data_by_indices

from threading import Thread

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    )
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Client(Thread):
    def __init__(self, name, host, port):
        Thread.__init__(self)
        self.name = name
        self.logger = self.setup_logger(self.name)
        self.host = host
        self.port = port

        self.accs = []
        self.losses = []

    def run(self):
        '''
        Necessary method to use threads. Calls all functions for the FedAvg as it can be seen in the code.
        '''
        self.sock = self.setup_socket(self.host, self.port)
        self.send(self.name)

        self.logger.info("Client is connected. Waiting for server to start...")

        self.data_name = self.receive()

        data_indices = self.receive()
        self.data = get_data_by_indices(self.data_name, True, data_indices)
        self.logger.debug(f"Received data from server.")

        config = self.receive()
        self.set_params(epochs=config["epochs"],
                        batch_size=config["batch_size"],
                        optimizer=config["optimizer"],
                        learning_rate=config["learning_rate"],
                        criterion=config["criterion"])
        self.logger.debug(f"Received parameters from server and set them.")
        self.logger.info(f"Data distribution: {str(self.class_distribution())}")

        self.model = self.receive()
        self.logger.debug(f"Received model from server.")

        self.logger.info("Ready to start!")
        while True:
            signal = self.receive()
            if not signal:
                self.logger.info("No data received: Exiting")
                break
            self.logger.info(f"{self.name} <--{signal}-- Server")
            if signal == "Update":
                model = self.receive()
                self.model.load_state_dict(model)
                self.logger.debug(f"{self.name} <--Model-- Server")
                self.update()
                self.send(self.model.state_dict())
                self.logger.debug(f"{self.name} --Model--> Server")
            elif signal == "Skip":
                self.send(self.model.state_dict())
                self.logger.debug(f"{self.name} --Model--> Server")
            elif signal == "Finish":
                model = self.receive()
                self.model.load_state_dict(model)
                self.logger.debug(f"{self.name} <--Complete model-- Server")

        # Plot performance
        fig, ax = plt.subplots()
        ax.plot(self.accs)
        ax.plot(self.losses)
        ax.set_xticklabels(np.arange(1, self.epochs + 1, dtype="int32").tolist() * int(len(self.accs)/self.epochs))
        ax.set_xticks(np.arange(0, len(self.accs)))
        ax.grid()
        ax.legend(["Accuracy", "Loss"])
        fig.savefig("latest_performance_" + self.name + ".png")


    def set_params(self, epochs, batch_size, optimizer, learning_rate, criterion):
        """
        Set parameters of the client for the training rounds. Must be called at least once after creation of a
        client.

        :param epochs: int, number of epochs the client will train
        :param batch_size: int, batch size for the training
        :param optimizer: function, optimizer for this client
        :param learning_rate: float
        :param criterion: function
        """
        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=batch_size, shuffle=True)
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.criterion = criterion

    def update(self):
        """
        Does one round of training for the specified number of epochs.
        """
        self.logger.debug(f"Start training...")

        self.model.train()
        

        optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            self.logger.debug(f"Epoch {epoch+1}/{self.epochs}...")
            for x, y in self.dataloader:
                optimizer.zero_grad()

                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                optimizer.step()
            loss, acc = self.evaluate()
            self.losses.append(loss)
            self.accs.append(acc)
            self.logger.info(f"Epoch {epoch+1}/{self.epochs} completed: loss: {loss}, accuracy: {acc}.")
        self.logger.info("Finished training!")

    def evaluate(self):
        """
        Evaluate the current client's model with its data

        :return: loss: loss of the model given its data
                 acc: accuracy of the model given its data
        """
        self.model.eval()
        loss = 0
        acc = 0

        with torch.no_grad():
            for x, y in self.dataloader:
                outputs = self.model(x)
                loss += self.criterion(outputs, y).item()
                preds = outputs.argmax(dim=1, keepdim=True)
                acc += (preds == y.view_as(preds)).sum().item()

        loss = loss / len(self.dataloader)
        acc = acc / len(self.data)

        return loss, acc

    def class_distribution(self):
        '''
        Calculates the number of instances per class and retunrs it.

        :return: Class distribution of the local dataset
        '''
        return sorted(Counter(self.data.dataset.targets[self.data.indices].numpy()).items())

    def setup_socket(self, host, port):
        sock = socket(AF_INET, SOCK_STREAM)
        sock.connect((host, port))
        return sock

    def setup_logger(self, name):
        logger = logging.getLogger(name)
        #logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        return logger

    def send(self, msg):
        '''
        Sends msg to the server

        :param msg: anything, data/msg to send to the client
        '''
        msg = pickle.dumps(msg)
        size = struct.pack("I", len(msg))
        self.sock.send(size + msg)

    def receive(self):
        '''
        Waits for data from the server and returns it.

        :return: data sent by the server
        '''
        size = self.sock.recv(4)
        if not size: return None
        size = struct.unpack("I", size)[0]
        data = bytearray()
        recv_bytes = 0
        buffer = 4096
        while recv_bytes < size:
            if (size - recv_bytes) < buffer: buffer = size - recv_bytes
            msg = self.sock.recv(buffer)
            data.extend(msg)
            recv_bytes += len(msg)
        return pickle.loads(data)

if __name__ == "__main__":
    client = Client("client", SERVER_HOST, SERVER_PORT)