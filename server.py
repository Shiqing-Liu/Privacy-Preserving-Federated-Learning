import torch
import copy, os, time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle
import struct
from socket import socket, AF_INET, SOCK_STREAM
import concurrent.futures

from threading import Thread, Lock

from config import SERVER_HOST, SERVER_PORT, SAVE_PATH, device
from utils import get_data, split_data_by_indices
from models import *
from matplotlib.ticker import MaxNLocator

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Server(Thread):
    def __init__(self, model, fed_config, host, port, lock):
        Thread.__init__(self)
        # Setting all required configurations
        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]
        self.ternary = fed_config["ternary"]

        self.optimizer = fed_config["optimizer"]
        self.criterion = fed_config["criterion"]
        self.learning_rate = fed_config["lr"]

        self.model = model
        self.data_name = fed_config["data_name"]
        self.iid = fed_config["iid"]
        self.shards_each = fed_config["shards_each"]
        self.personalized = fed_config["personalized"]
        self.clients = []
        self.clients_data_len = []
        self.clients_names = []

        self.cur_round = 1
        self.received_data = []
        self.send_data = []

        self.losses = []
        self.accs = []

        # Setup logger
        self.logger = self.setup_logger()

        self.host = host
        self.port = port
        self.lock = lock

    def run(self):
        '''
        Necessary method to use threads. Calls all functions for the FedAvg as it can be seen in the code (the names of
        the functions speak for themselves)
        '''
        # Setup Socket
        with self.lock:
            self.sock = self.setup_socket(self.host, self.port)

        # Start the server
        self.connect_clients()
        self.setup_data_and_model()
        self.fit()

        # Save results to file (Is done before disconnecting the clients to avoid that multiple threads access the file)
        with open(os.path.join(SAVE_PATH, "configuration.txt"), 'a') as f:
            f.write(f"Information from Server:\n\n")
            f.write(f"Accuracy: {self.accs}\n")
            f.write(f"Loss: {self.losses}\n")
            f.write(f"Received data: {self.received_data}\n")
            f.write(f"Send data: {self.send_data}\n\n\n")

        # End
        self.disconnect_clients()
        self.logger.info("Exiting.")
        self.logger.debug(f"Received bytes = {self.received_data}; transmitted bytes = {self.received_data}")
        with self.lock:
            # Plot performance
            fig, ax = plt.subplots()
            ax.plot(list(range(len(self.losses))), self.losses, color='blue')
            ax.set_xlabel("Global Rounds")
            ax.set_ylabel('Loss')
            ax.legend(["Loss"], loc="center left")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            ax2 = ax.twinx()
            ax2.plot(list(range(len(self.accs))), self.accs, color='orange')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim([-0.05, 1.05])
            ax2.legend(["Accuracy"], loc="center right")
            ax.grid()

            plt.title(f"Server Performance")
            fig.savefig(os.path.join(SAVE_PATH, "performance_server.png"))

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
        with self.lock:
            # Plot Data
            fig, ax = plt.subplots()
            ax.plot(list(cumsum_rec.keys()), np.cumsum((list(cumsum_rec.values()))), "--")
            ax.plot(list(cumsum_send.keys()), np.cumsum((list(cumsum_send.values()))), "-.")
            plt.xlabel("Rounds")
            plt.ylabel("Bytesleistung")
            plt.title(f"Server send/receive")
            ax.grid()
            ax.legend(["Received Bytes", "Send Bytes"])
            fig.savefig(os.path.join(SAVE_PATH, "received_and_transmitted_server.png"))

    def connect_clients(self):
        '''
        Wait for the correct number of clients (as specified in self.num_clients) to connect.
        '''
        self.logger.info(f"Waiting for {self.num_clients} clients to connect...")
        while len(self.clients) != self.num_clients:
            conn, addr = self.sock.accept()
            self.clients.append((conn, addr))
            self.clients_names.append(self.receive(conn, addr))
            self.logger.info(f"New connection {len(self.clients)}/{self.num_clients}: {addr[1]}, {self.clients_names[-1]}")

        self.logger.debug("All required clients are connected!")

    def disconnect_clients(self):
        '''
        Closes the connections to the clients.
        '''
        for conn, addr in self.clients:
            conn.close()
            self.logger.info(f"Closed connection to {addr[1]}")

    def setup_data_and_model(self):
        """
        Setup server and clients. This method gets the test data, splits the train data indices and sends everything
        important to the clients (data name, indices, config, model).
        """
        self.logger.debug("Start data & model setup.")

        self.data = get_data(self.data_name, train=True)
        self.test_data, self.train_data = get_data(self.data_name, train=False)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

        splits = split_data_by_indices(self.data, self.num_clients, iid=self.iid, shards_each=self.shards_each)

        client_config = {"epochs": self.local_epochs,
                         "batch_size": self.batch_size,
                         "optimizer": self.optimizer,
                         "learning_rate": self.learning_rate,
                         "criterion": self.criterion,
                         "data_name": self.data_name,
                         "ternary": self.ternary,
                         "personalized": self.personalized}
        for i, (conn, addr) in enumerate(self.clients):
            self.send(conn, addr, self.data_name)
            self.clients_data_len.append(len(splits[i]))
            self.send(conn, addr, splits[i])
            self.send(conn, addr, client_config)
            self.send(conn, addr, copy.deepcopy(self.model))

        self.logger.info("Data & model setup finished.")

    def average_model(self, client_models, coefficients):
        """
        Average the global model with the clients models. These will be weighted according to the coefficients.

        :param client_models: State dicts of the clients models in a list
        :param coefficients: Coefficients in a list
        """
      
        global_dict = self.model.state_dict()
        averaged_dict = OrderedDict()
        for layer in global_dict.keys():
            averaged_dict[layer] = torch.zeros(global_dict[layer].shape, dtype=torch.float32)
            for client_dict, coef in zip(client_models, coefficients):
                client_dict[layer] = client_dict[layer].to("cpu")
                averaged_dict[layer] += coef * client_dict[layer]

        self.model.load_state_dict(averaged_dict)

    def average_model_quant(self, client_models, coefficients):

        # number of clients participatiuon
        num_model = len(client_models)
        # weight the weights by client contribution for strategy 2
        w_avg = client_models[0]
        for key, value in w_avg.items():
            for i in range(0, num_model):
                if i == 0:
                    w_avg[key] = coefficients[0] * client_models[0][key]
                else:
                    w_avg[key] += coefficients[i] * client_models[i][key]
        # copy and quantizise for strategy 1
        backup_w = copy.deepcopy(w_avg)

        ter_avg = self.quantize_server(backup_w)
        w,_ = self.choose_model(w_avg,ter_avg)
        self.model.load_state_dict(w)

    def quantize_server(self, model_dict):
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
        #tmp_net1 = nn.Sequential(*list(tmp_net1.modules())[:-1])
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

    def train(self):
        """
        One round of training. Consisting of choosing the fraction of clients involved in this round, updating the
        clients models, collecting the updates and averaging the global model.
        """
        m = np.maximum(int(self.fraction * len(self.clients)), 1)
        indices = np.random.choice(self.num_clients, m, replace=False)

        signals = ["Skip"] * self.num_clients
        list(map(signals.__setitem__, indices, ["Update"] * len(indices)))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for signal, (conn, addr), name in zip(signals, self.clients, self.clients_names):
                futures.append(executor.submit(self.send_signal, signal, conn, addr, name))

            finished_round = False
            while not finished_round:
                finished_round = True
                for future in futures:
                    if not future.done():
                        finished_round = False
                        continue
            client_models = [future.result() for future in futures]  #last layer should be filled with random numbers



        coefficients = [size/sum(self.clients_data_len) for size in self.clients_data_len]
        if self.ternary:
            self.average_model_quant(client_models, coefficients)
        else:
            self.average_model(client_models,coefficients)

    def fit(self):
        """
        Starts the training for the specified number of rounds.
        """
        self.logger.debug("Start training...")

        for r in range(1, self.num_rounds + 1):
            start = time.time()
            self.cur_round += 1
            self.logger.debug(f"Round {r}/{self.num_rounds}...")
            self.train()
            loss, acc = self.evaluate()
            self.losses.append(loss)
            self.accs.append(acc)
            dur = time.time() - start
            self.logger.info(f"Round {r}/{self.num_rounds} completed ({int(dur//60)} min {int(dur%60)} sec): loss: {loss:.3f}, accuracy: {acc:.3f}.")

        for (conn, addr), name in zip(self.clients, self.clients_names):
            self.send_signal("Finish", conn, addr, name)

        self.logger.info("Finished training!")

    def train_personalized_layer(self):
        '''
        Trains the LAST layer of the model, all other layers are freezed.
        '''
        self.model = self.model.to(device)
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
        list(self.model.parameters())[-1].requires_grad = True

        optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        num_epochs = 5
        for epoch in range(num_epochs):
            start = time.time()
            self.logger.debug(f"Train personalised layer: Epoch {epoch + 1}/{num_epochs}...")
            for x, y in self.train_dataloader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                optimizer.step()
        loss, acc = self.evaluate()
        self.logger.info(f"Train personalised layer: Epoch {num_epochs}/{num_epochs} completed ({time.time()-start} sec)| loss: {loss} | accuracy: {acc}.")

    def evaluate(self, eval_model = None):
        """
        Evaluate the current global model with the specified test data.

        :return: loss: loss of the model given its data
        :return: acc : accuracy of the model given its data
        """
        if eval_model is None:
            eval_model = self.model
        eval_model = eval_model.to(device)
        eval_model.eval()
        loss = 0
        acc = 0
        with torch.no_grad():
            for x, y in self.test_dataloader:
                x = x.to(device)
                y = y.to(device)
                outputs = eval_model(x)
                loss += self.criterion(outputs, y).item()
                preds = outputs.argmax(dim=1, keepdim=True)
                acc += (preds == y.view_as(preds)).sum().item()
        loss = loss / len(self.test_dataloader)
        acc = acc / len(self.test_data)

        return loss, acc

    def setup_logger(self, name="Server"):
        logger = logging.getLogger(name)
        #logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        return logger

    def setup_socket(self, host, port):
        sock = socket(AF_INET, SOCK_STREAM)
        sock.bind((host, port))
        sock.listen()
        return sock

    def send(self,conn, addr, msg):
        '''
        Sends msg to the specified client.

        :param conn: conn of the client
        :param addr: addr of the client
        :param msg: data/msg to send to the client
        '''
        msg = pickle.dumps(msg)
        size = struct.pack("I", len(msg))
        self.send_data.append((self.cur_round, len(msg)))
        conn.send(size + msg)

    def receive(self, conn, addr):
        '''
        Waits for data of the specified client and returns it.

        :param conn: conn of the client
        :param addr: addr of the client
        :return: data sent by this client
        '''
        size = conn.recv(4)
        size = struct.unpack("I", size)[0]
        data = bytearray()
        recv_bytes = 0
        buffer = 4096
        while recv_bytes < size:
            if (size - recv_bytes) < buffer: buffer = size - recv_bytes
            msg = conn.recv(buffer)
            data.extend(msg)
            recv_bytes += len(msg)
        self.received_data.append((self.cur_round, len(data)))
        return pickle.loads(data)

    def send_signal(self, signal, conn, addr, name): # Signal is Update, Skip or Finish
        '''
        Sends signal to the client. Depending on the signal the following will happen:
            a) "Update": Sends global model, waits for local model
            b) "Skip": Waits for local model
            c) "Finish": Sends global mode, waits for local model
        :param signal: A string that will be send to the client. One from the above
        :param conn: conn of client
        :param addr: addr of client
        :param name: name of the client for logging
        '''
        self.send(conn, addr, signal)
        self.logger.debug(f"Server --{signal}--> {name}")
        if signal == "Update":
            self.send(conn, addr, self.model.state_dict())
            self.logger.debug(f"Server --Model--> {name}")
        elif signal == "Finish":
            self.send(conn, addr, self.model.state_dict())
            self.logger.debug(f"Server --Model--> {name}")
            return
        data = self.receive(conn, addr)
        self.logger.debug(f"Server <--Model-- {name}")
        return data

if __name__ == "__main__":
    fed_config = {"C": 0.2,
                  "K": 2,
                  "R": 2,
                  "E": 1,
                  "B": 64,
                  "optimizer": torch.optim.SGD,
                  "criterion": nn.CrossEntropyLoss(),
                  "lr": 0.01,
                  "data_name": "MNIST",
                  "iid": False,
                  "shards_each": 2}
    model = Net_2()
    server = Server(model, fed_config, SERVER_HOST, SERVER_PORT)
