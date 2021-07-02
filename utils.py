import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users



def mnist_nid():
    pass


def cifar_iid():
    pass


from torchvision import datasets, transforms
trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
mnist_iid(dataset_train,10)
