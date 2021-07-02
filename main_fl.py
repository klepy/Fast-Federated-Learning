# Python version: 3.8.8 
# venv: CE
# Editor : klepy

import argparse
from typing_extensions import ParamSpec
import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
from torchvision import datasets, transforms

from utils import mnist_iid, mnist_nid, cifar_iid
from update import LocalUpdate, GlobalUpdate
from models import FNN, VGG16
from average import FedAvg, UniformAvg 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic Federated Learning')
    # Federated arguments
    parser.add_argument('--rounds', type=int, default=10, help="rounds of training: K")
    parser.add_argument('--num_users', type=int, default=100, help="number of clients: M")
    parser.add_argument('--frac', type=float, default=0.1, help='fraction of clients: C')
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default:0.9)")
    parser.add_argument('--model', type=str, default='FNN', help="Model Type (FNN or VGG16)")
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default="mnist", help="name of dataset: mnist/cifar10")
    parser.add_argument('--iid', type=int, default=1, help="Enter i.i.d(1) / non i.i.d (0)")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()

    # Load Data and Split
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_nid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape

    # Create Model to train
    if args.model == 'FNN' and args.dataset == 'mnist':
        net_glob = FNN().cuda()
    elif args.model == 'VGG16' and args.dataset =='cifar10':
        net_glob = VGG16().cuda()
    else:
        print(f'Check your train options')
    
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training 
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.rounds):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))