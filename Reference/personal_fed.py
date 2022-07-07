import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torchsummary import summary
import time
import random
import logging
import json
from hashlib import md5
import copy
import easydict
import os
import sys
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import pickle
import dill

# Directory where the json file of arguments will be present
directory = './Parse_Files'


# Import different files required for loading dataset, model, testing, training
from utility.LoadSplit import Load_Dataset, Load_Model
# from utility.options import args_parser
from models.Update import train_client,test_client,finetune_client
from models.Fed import FedAvg
from models.test import test_img

torch.manual_seed(0)


if __name__ == '__main__':
    
    # Initialize argument dictionary
    args = {}
    
    
    # From Parse_Files folder, get the name of required parse file which is provided while running this script from bash
    f = directory+'/'+str(sys.argv[1])
    print(f)
    with open(f) as json_file:  
        args = json.load(json_file)

        
    # Taking hash of config values and using it as filename for storing model parameters and logs
    param_str = json.dumps(args)
    file_name = md5(param_str.encode()).hexdigest()

    
    # Converting args to easydict to access elements as args.device rather than args[device]
    args = easydict.EasyDict(args)
    print(args)


    # Save configurations by making a file using hash value
    with open('./config/parser_{}.txt'.format(file_name),'w') as outfile:
        json.dump(args,outfile,indent=4)

    SUMMARY = os.path.join('./results',file_name)
    args.summary=SUMMARY
    os.makedirs(SUMMARY)
        
    # Setting the device - GPU or CPU
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # net_glob = Load_Model(args=args) 
    # print(net_glob) 
    # exit()
    # Load the training and testing datasets
    dataset_train, dataset_test, dict_users = Load_Dataset(args=args)

    
    # Initialize Global Server Model
    net_glob = Load_Model(args=args) 
    # print(net_glob)   
    net_glob.train()

    # Print name of the architecture - 'MobileNet or ResNet or NewNet'
    print(args.model)

    
    # copy weights
    w_glob = net_glob.state_dict()

    
    # Set up log file
    logging.basicConfig(filename='./log/{}.log'.format(file_name),format='%(message)s',level=logging.DEBUG)

    tree=lambda : defaultdict(tree)
    stats=tree()
    writer = SummaryWriter(args.summary)

    #splitting user data into training and testing parts
    train_data_users = {}
    test_data_users = {}

    for i in range(args.num_users):
        dict_users[i] = list(dict_users[i])
        train_data_users[i] = list(random.sample(dict_users[i],int(args.split_ratio*len(dict_users[i]))))
        test_data_users[i] = list(set(dict_users[i])-set(train_data_users[i]))

    # exit()
    # local models for each client
    local_nets = {}

    for i in range(0,args.num_users):
        local_nets[i] = Load_Model(args=args)
        local_nets[i].train()
        local_nets[i].load_state_dict(w_glob)


    # Start training

    logging.info("Training")

    start = time.time()

    for iter in range(args.epochs):
        
        print('Round {}'.format(iter))

        logging.info("---------Round {}---------".format(iter))

        w_locals, loss_locals = [], []

        for idx in range(0,args.num_users):

            w,loss = train_client(args,dataset_train,train_data_users[idx],net = local_nets[idx])
            w_locals.append(w)
            loss_locals.append(copy.deepcopy(loss))

        # store testing and training accuracies of the model before global aggregation
        logging.info("Testing Client Models before aggregation")
        logging.info("")
        s = 0
        for i in range(args.num_users):
            logging.info("Client {}:".format(i))
            acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],local_nets[i])
            acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],local_nets[i])
            logging.info("Training accuracy: {:.3f}".format(acc_train))
            logging.info("Testing accuracy: {:.3f}".format(acc_test))
            logging.info("")
            # print(acc_test)
            stats[i][iter]['Before Training accuracy']=acc_train
            stats[i][iter]['Before Test accuracy']=acc_test
            writer.add_scalar(str(i)+'/Before Training accuracy',acc_train,iter)
            writer.add_scalar(str(i)+'/Before Test accuracy',acc_test,iter)
            
            s += acc_test
        s /= args.num_users
        logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
        stats['Before Average'][iter]=s
        writer.add_scalar('Average'+'/Before Test accuracy',s,iter)

        # hyperparameter = number of layers we want to keep in the base part
        base_layers = args.base_layers

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # Updating base layers of the clients and keeping the personalized layers same
        for idx in range(args.num_users):
            for i  in list(w_glob.keys())[0:base_layers]:
                w_locals[idx][i] = copy.deepcopy(w_glob[i]) 
            local_nets[idx].load_state_dict(w_locals[idx])

        # store train and test accuracies after updating local models
        logging.info("Testing Client Models after aggregation")
        logging.info("")
        s = 0
        for i in range(args.num_users):
            logging.info("Client {}:".format(i))
            acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],local_nets[i])
            acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],local_nets[i])
            logging.info("Training accuracy: {:.3f}".format(acc_train))
            logging.info("Testing accuracy: {:.3f}".format(acc_test))
            logging.info("")

            stats[i][iter]['After Training accuracy']=acc_train
            stats[i][iter]['After Test accuracy']=acc_test
            writer.add_scalar(str(i)+'/After Training accuracy',acc_train,iter)
            writer.add_scalar(str(i)+'/After Test accuracy',acc_test,iter)

            s += acc_test
        s /= args.num_users
        logging.info("Average Client accuracy on their test data: {: .3f}".format(s))

        stats['After Average'][iter]=s
        writer.add_scalar('Average'+'/After Test accuracy',s,iter)


        loss_avg = sum(loss_locals) / len(loss_locals)
        logging.info('Average loss of clients: {:.3f}'.format(loss_avg))


        ###FineTuning
        if args.finetune:
            # print("FineTuning")
            personal_params=list(w_glob.keys())[base_layers:]
            for idx in range(0,args.num_users):
                for i,param in enumerate(local_nets[idx].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad=False
                w,loss = finetune_client(args,dataset_train,train_data_users[idx],net = local_nets[idx])
                for i,param in enumerate(local_nets[idx].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad=True

            s = 0
            for i in range(args.num_users):
                logging.info("Client {}:".format(i))
                acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],local_nets[i])
                acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],local_nets[i])
                logging.info("Training accuracy after finetune: {:.3f}".format(acc_train))
                logging.info("Testing accuracy after finetune: {:.3f}".format(acc_test))
                logging.info("")

                stats[i][iter]['After finetune Training accuracy']=acc_train
                stats[i][iter]['After finetune Test accuracy']=acc_test
                writer.add_scalar(str(i)+'/After finetune Training accuracy',acc_train,iter)
                writer.add_scalar(str(i)+'/After finetune Test accuracy',acc_test,iter)

                s += acc_test
            s /= args.num_users
            logging.info("Average Client accuracy on their test data: {: .3f}".format(s))

            stats['After finetune Average'][iter]=s

    end = time.time()

    logging.info("Training Time: {}s".format(end-start))
    logging.info("End of Training")

    # save model parameters
    torch.save(net_glob.state_dict(),'./state_dict/server_{}.pt'.format(file_name))
    for i in range(args.num_users):
        torch.save(local_nets[i].state_dict(),'./state_dict/client_{}_{}.pt'.format(i,file_name))


    # test global model on training set and testing set

    logging.info("")
    logging.info("Testing")

    logging.info("Global Server Model")
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    logging.info("Training accuracy of Server: {:.3f}".format(acc_train))
    logging.info("Training loss of Server: {:.3f}".format(loss_train))
    logging.info("Testing accuracy of Server: {:.3f}".format(acc_test))
    logging.info("Testing loss of Server: {:.3f}".format(loss_test))
    logging.info("End of Server Model Testing")
    logging.info("")

    logging.info("Client Models")
    s = 0
    # testing local models
    for i in range(args.num_users):
        logging.info("Client {}:".format(i))
        acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],local_nets[i])
        acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],local_nets[i])
        logging.info("Training accuracy: {:.3f}".format(acc_train))
        logging.info("Training loss: {:.3f}".format(loss_train))
        logging.info("Testing accuracy: {:.3f}".format(acc_test))
        logging.info("Testing loss: {:.3f}".format(loss_test))
        logging.info("")
        s += acc_test
    s /= args.num_users
    logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    logging.info("End of Client Model testing")

    
    logging.info("")
    logging.info("Testing global model on individual client's test data")
    
    # testing global model on individual client's test data
    s = 0
    for i in range(args.num_users):
        logging.info("Client {}".format(i))
        acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],net_glob)
        acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],net_glob)
        logging.info("Training accuracy: {:.3f}".format(acc_train))
        logging.info("Testing accuracy: {:.3f}".format(acc_test))
        s += acc_test
    s /= args.num_users
    logging.info("Average Client accuracy of global model on each client's test data: {: .3f}".format(s))

    dill.dump(stats,open(os.path.join(args.summary,'stats.pkl'),'wb'))
    writer.close()
    # print(stats['After Average'])
    # print(stats['After finetune Average'])