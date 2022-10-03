import torch
import torchvision
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split


LESS_DATA = 2000 # Int, >0 if less data should be used, otherwise 0
SERVER_TEST_SIZE = 1000
SERVER_TRAIN_SIZE = 100


def get_data_by_indices(name, train, indices):
    '''
    Returns the data of the indices.

    :param name: string, name of the dataset
    :param train: boolean, train or test data
    :param indices: list, indices of the data to use
    :return: dataset with the given indices
    '''
    if name == "CIFAR10": # 10 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif name == "MNIST": # 10 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    elif name == "FashionMNIST": # 10 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    elif name == "CIFAR100": # 100 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    else:
        raise NameError(f"No dataset named {name}. Choose from: CIFAR10, CIFAR100, MNIST, FashionMNIST")

    return torch.utils.data.Subset(dataset, indices)

def split_data_by_indices(data, n, iid=True, shards_each=2):
    '''
    Splits the given data in n splits, instances are represented by their indices. Splits can be either idd or not-iid, for the latter the parameter shards_each
    will be used as well meaning the data will be sorted by class, splited in shards_each * n splits and #shards_each
    will be randomly assigned to each client

    :param data: Dataset
    :param n: int, number of splits
    :param iid: boolean, iid or non-iid splits
    :param shards_each: int, see description
    :return: list, containing the splits as indices
    '''
    data_size = LESS_DATA if LESS_DATA > 0 else len(data)

    if iid:
        local_len = np.floor(data_size / n)
        total_n = int(local_len * n)

        indices = torch.randperm(len(data)).numpy()[:total_n]
        splits = np.split(indices, n)

    else:
        n_shards = n * shards_each
        shard_len = int(np.floor(data_size / n_shards))

        indices = torch.randperm(len(data)).numpy()[:(n_shards * shard_len)]
        targets = torch.Tensor(data.targets)
        ind_targets = targets[indices]

        sorted_indices = np.array([x for _, x in sorted(zip(ind_targets, indices))])
        shards = np.split(sorted_indices, n_shards)

        random_shards = torch.randperm(n_shards)
        shards = [shards[i] for i in random_shards]

        splits = []
        for i in range(0, len(random_shards), shards_each):
            splits.append(np.concatenate(shards[i:i + shards_each]).tolist())

    return splits

def split_data(data, n, iid=True, shards_each=2):
    '''
    Does the same like split_data_by_indices but returns the data directly instead of indices.

    :param data: Dataset
    :param n: int, number of splits
    :param iid: boolean, iid or non-iid splits
    :param shards_each: int, see description
    :return: list, containing the splits
    '''
    local_len = np.floor(len(data) / n)
    total_n = int(local_len * n)

    if iid:
        indices = torch.randperm(len(data))[:total_n]
        subset = torch.utils.data.Subset(data, indices)
        splits = torch.utils.data.random_split(subset, np.full(n, fill_value=local_len, dtype="int32"))

    else:
        n_shards = n * shards_each

        sorted_indices = torch.argsort(data.targets)

        shard_len = int(np.floor(len(data) / n_shards))
        shards = list(torch.split(sorted_indices, shard_len))[:n_shards]

        random_shards = torch.randperm(n_shards)
        shards = [shards[i] for i in random_shards]

        splits = []
        for i in range(0, len(random_shards), shards_each):
            splits.append(torch.utils.data.Subset(data, torch.cat(shards[i:i + shards_each])))

    return splits

def get_data(name, train):
    '''
    Gets the corresponding data (either train or test data)
    :param name: string, name of the dataset
    :param train: boolean, train or test
    :return: dataset
    '''
    if name == "CIFAR10": # 10 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif name == "MNIST": # 10 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    elif name == "FashionMNIST": # 10 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    elif name == "CIFAR100": # 100 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    else:
        raise NameError(f"No dataset named {name}. Choose from: CIFAR10, CIFAR100, MNIST, FashionMNIST")

    if not train:
        indices = list(zip(np.arange(len(dataset)), dataset.targets))
        test_indices, train_indices = train_test_split(indices, test_size=SERVER_TEST_SIZE, train_size=SERVER_TRAIN_SIZE, shuffle=True, stratify=dataset.targets)
        test_indices, _ = zip(*test_indices)
        train_indices, _ = zip(*train_indices)
        dataset = (torch.utils.data.Subset(dataset, test_indices), torch.utils.data.Subset(dataset, train_indices))



    return dataset

class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

if __name__ == '__main__':
    data = get_data("CIFAR10", False)