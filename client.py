import os

import torch
from collections import Counter
import pickle
import struct
import copy
from socket import socket, AF_INET, SOCK_STREAM
import matplotlib.pyplot as plt
import numpy as np
from config import SERVER_HOST, SERVER_PORT, SAVE_PATH
from utils import get_data_by_indices
from threading import Thread
import logging
from random import random
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Client(Thread):
    def __init__(self, name, host, port, lock):
        Thread.__init__(self)
        self.name = name
        self.logger = self.setup_logger(self.name)
        self.host = host
        self.port = port
        self.send_data = []
        self.received_data = []
        self.strategy_history = []
        self.accs = []
        self.losses = []
        self.signals = []
        self.personalized_weight = []
        self.lock = lock
        self.round = 0

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
                        criterion=config["criterion"],
                        ternary=config["ternary"],
                        personalized=config["personalized"])
        self.logger.debug(f"Received parameters from server and set them.")
        self.logger.info(f"Data distribution: {str(self.class_distribution())}")

        self.model = self.receive()
        self.logger.debug(f"Received model from server.")

        self.logger.info("Ready to start!")
        while True:
            signal = self.receive()
            self.signals.append(signal)
            if not signal:
                self.logger.info("No data received: Exiting")
                break
            self.logger.info(f"{self.name} <--{signal}-- Server")
            if signal == "Update":
                model = self.receive()
                self.model.load_state_dict(model)
                self.logger.debug(f"{self.name} <--Model-- Server")
                # list of keys to acess the weights of last layer
                list_keys_weights = list(self.model.state_dict().keys())

                if not self.personalized:
                    self.update()
                    self.send(self.model.state_dict())

                else:
                    if len(self.personalized_weight) == 0:         #first time the client recieves data from the server
                        self.update()
                    else:
                        self.model.state_dict().update(self.personalized_weight[-1])    #the client gets the last saved values of the last layer's wights again so the weights of the last layer received from the server don't matter
                        self.update()

                    # weights + bias of last layer after the update
                    personalized_weights = {list_keys_weights[k]: self.model.state_dict()[list_keys_weights[k]] for k in (-2,-1)}
                    self.personalized_weight.append(personalized_weights)

                    #replace last layer's weights and bias with random numbers before sending them to the server
                    fc_weight_to_np = self.model.state_dict()[list_keys_weights[-2]].cpu().detach().numpy()
                    fc_bias_to_np = self.model.state_dict()[list_keys_weights[-1]].cpu().detach().numpy()
                    for index in np.ndindex(fc_bias_to_np.shape):
                        fc_bias_to_np[index] = random()
                    fc_bias_to_tensor = torch.from_numpy(fc_bias_to_np)

                    for index in np.ndindex(fc_weight_to_np.shape):
                        fc_weight_to_np[index] = random()
                    fc_weight_to_tensor = torch.from_numpy(fc_weight_to_np)

                    last_layer_to_be_sent = {list_keys_weights[-2]: fc_weight_to_tensor, list_keys_weights[-1]: fc_bias_to_tensor}
                    self.model.state_dict().update(last_layer_to_be_sent)

                    #send weights to the server
                    self.send(self.model.state_dict())

                    # personalized weights get their values again
                    last_weights = self.personalized_weight[-1]
                    self.model.state_dict().update(last_weights)

                self.logger.debug(f"{self.name} --Model--> Server")
            elif signal == "Skip":
                if self.personalized:
                    # list of keys to acess the weights of last layer
                    list_keys_weights = list(self.model.state_dict().keys())
                    # weights + bias of last layer after the update
                    personalized_weights = {list_keys_weights[k]: self.model.state_dict()[list_keys_weights[k]] for k in
                                            (-2, -1)}
                    self.personalized_weight.append(personalized_weights)

                    # replace last layer's weights and bias with random numbers before sending them to the server
                    fc_weight_to_np = self.model.state_dict()[list_keys_weights[-2]].cpu().detach().numpy()
                    fc_bias_to_np = self.model.state_dict()[list_keys_weights[-1]].cpu().detach().numpy()
                    for index in np.ndindex(fc_bias_to_np.shape):
                        fc_bias_to_np[index] = random()
                    fc_bias_to_tensor = torch.from_numpy(fc_bias_to_np)

                    for index in np.ndindex(fc_weight_to_np.shape):
                        fc_weight_to_np[index] = random()
                    fc_weight_to_tensor = torch.from_numpy(fc_weight_to_np)

                    last_layer_to_be_sent = {list_keys_weights[-2]: fc_weight_to_tensor,
                                            list_keys_weights[-1]: fc_bias_to_tensor}
                    self.model.state_dict().update(last_layer_to_be_sent)

                    # send weights to the server
                    self.send(self.model.state_dict())

                    # personalized weights get their values again
                    last_weights = self.personalized_weight[-1]
                    self.model.state_dict().update(last_weights)
                else:
                    self.send(self.model.state_dict())

                self.logger.debug(f"{self.name} --Model--> Server")
            elif signal == "Finish":
                model = self.receive()
                self.model.load_state_dict(model)
                self.logger.debug(f"{self.name} <--Complete model-- Server")
        self.logger.debug(f"Received bytes = {self.received_data}; transmitted bytes = {self.received_data}")

        # Plot performance
        fig, ax = plt.subplots()
        ax.plot(self.losses)
        ax.set_xticklabels(np.arange(1, self.epochs + 1, dtype="int32").tolist() * int(len(self.accs)/self.epochs))
        ax.set_xticks(np.arange(0, len(self.accs)))
        ax.set_ylabel('Loss')
        ax2 = ax.twinx()
        plt.ylabel('Accuracy')
        ax2.plot(self.accs)

        ax.grid()
        plt.title(f"{self.name} performance")
        ax.legend(["Accuracy", "Loss"])
        fig.savefig(os.path.join(SAVE_PATH, "performance_" + self.name + ".png"))

        # Save results to file
        with self.lock:
            with open(os.path.join(SAVE_PATH, "configuration.txt"), 'a') as f:
                f.write(f"Information from {self.name}:\n\n")
                f.write(f"Signals: {self.signals}\n")
                f.write(f"Data distribution: {self.class_distribution()}\n")
                f.write(f"Accuracy: {self.accs}\n")
                f.write(f"Loss: {self.losses}\n")
                f.write(f"Received data: {self.received_data}\n")
                f.write(f"Send data: {self.send_data}\n")
                f.write(f"Strategies (if used): {self.strategy_history}\n\n\n")
        cumsum_send = {}
        for (i, j) in self.send_data:
            if i in cumsum_send.keys():
                cumsum_send[i] += j
            else:
                cumsum_send[i] = j
        cumsum_rec = {}
        for (i, j) in self.received_data:
            if i in cumsum_rec.keys():
                cumsum_rec[i] += j
            else:
                cumsum_rec[i] = j

        # Plot Data
        fig, ax = plt.subplots()
        ax.plot(list(cumsum_rec.keys()), np.cumsum((list(cumsum_rec.values()))), "--")
        ax.plot(list(cumsum_send.keys()), np.cumsum((list(cumsum_send.values()))), "-.")
        plt.xlabel("Rounds")
        plt.title(f"{self.name} send/receive")
        plt.ylabel("Bytesleistung")
        ax.grid()
        ax.legend(["Received Bytes", "Send Bytes"])
        fig.savefig(os.path.join(SAVE_PATH, "send_and_transmit_" + self.name + ".png"))

    def set_params(self, epochs, batch_size, optimizer, learning_rate, criterion, ternary, personalized):
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
        self.ternary = ternary
        self.personalized = personalized

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
        if self.ternary:
            backup_w = self.model.state_dict().copy()
            ter_avg = self.quantize_client(backup_w)
            w, flag = self.choose_model(self.model.state_dict(), ter_avg)
            self.strategy_history.append("Strategy 1") if flag else self.strategy_history.append("Strategy 2")
            self.model.load_state_dict(w)
        self.round += 1
        self.logger.info("Finished training!")

    def quantize_client(self, model_dict):
        for key, kernel in model_dict.items():
            if 'ternary' and 'conv' in key:
                d2 = kernel.size(0) * kernel.size(1)
                delta = 0.05 * kernel.abs().sum() / d2
                tmp1 = (kernel.abs() > delta).sum()
                tmp2 = ((kernel.abs() > delta) * kernel.abs()).sum()
                w_p = tmp2 / tmp1
                a = (kernel > delta).float()
                b = (kernel < -delta).float()
                kernel = w_p * a - w_p * b
                model_dict[key] = kernel
        return model_dict

    def choose_model(self, f_dict, ter_dict):
        # create models based on both full and ternary weights
        tmp_net1 = copy.deepcopy(self.model)
        tmp_net2 = copy.deepcopy(self.model)
        tmp_net1.load_state_dict(f_dict)
        tmp_net2.load_state_dict(ter_dict)
        # evaluate networks on test set
        _, acc_1 = self.evaluate(tmp_net1)
        _, acc_2 = self.evaluate(tmp_net2)
        print('F: %.3f' % acc_1, 'TF: %.3f' % acc_2)
        # If the ter model loses more than 3 percent accuracy, sent full model instead
        flag = False
        if np.abs(acc_1 - acc_2) < 0.03:
            self.logger.info(f"Accuracy differnce: {np.abs(acc_1 - acc_2)}, Choosing Strategy 1")
            flag = True
            return ter_dict, flag
        else:
            self.logger.info(f"Accuracy differnce: {np.abs(acc_1 - acc_2)}, Choosing Strategy 2")
            return f_dict, flag

    def evaluate(self, eval_model=None):
        """
        Evaluate the current client's model with its data

        :return: loss: loss of the model given its data
                 acc: accuracy of the model given its data
        """
        if eval_model is None:
            eval_model = self.model
        eval_model.eval()

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
        np_targets = np.array(self.data.dataset.targets)
        return sorted(Counter(np_targets[self.data.indices]).items())

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
        self.send_data.append((self.round, len(msg)))
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
        self.received_data.append((self.round, len(data)))
        return pickle.loads(data)

if __name__ == "__main__":
    client = Client("client", SERVER_HOST, SERVER_PORT)
