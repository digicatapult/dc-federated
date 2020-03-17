import numpy as np

import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class MNISTNet(nn.Module):
    """
    CNN for the MNIST dataset.
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MNISTNetArgs(object):
    """
    Class to abstract the arguments for the MNIST model.
    """

    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 14
        self.lr = 1
        self.gamma = 0.7
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.save_model = False

    def print(self):
        print(f"batch_size: {self.batch_size}")
        print(f"test_batch_size: {self.test_batch_size}")
        print(f"epochs: {self.epochs}")
        print(f"lr: {self.lr}")
        print(f"gamma: {self.gamma}")
        print(f"no_cuda: {self.no_cuda}")
        print(f"seed: {self.seed}")
        print(f"log_interval: {self.log_interval}")
        print(f"save_model: {self.save_model}")


class MNISTSubSet(torch.utils.data.Dataset):
    """
    Represents a MNIST subset.

    Parameters
    ----------

    mnist_ds: torchvision.MNIST
        The core mnist dataset object.

    digits: int list
        The digits to restrict the data subset to
    """

    def __init__(self, mnist_ds, digits, args=None, transform=None, target_transform=None):

        self.digits = digits
        self.args = args
        mask = np.isin(mnist_ds.targets, digits)
        self.data = mnist_ds.data[mask].clone()
        self.targets = mnist_ds.targets[mask].clone()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None: img = self.transform(img)
        if self.target_transform is not None: target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_data_loader(self, use_cuda=False, kwargs=None):
        if kwargs is None:
            kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        return torch.utils.data.DataLoader(
            self, batch_size=self.args.batch_size, shuffle=True, **kwargs)


class MNISTModelTrainer(object):
    """
    Trainer for the MNIST data.

    Parameters
    ----------

    args: MNISTNetArgs (default None)
        The arguments for the MNISTNet

    model: MNISTNet (default None)
        The model for training.
    """
    def __init__(self, args=None, model=None):
        args = MNISTNetArgs() if not args else args
        self.use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        model = MNISTNet().to(self.device) if not model else model
        self.args = args
        self.model = model

    def train(self, train_loader, optimizer, epoch, batches_per_iter=None):
        """
        Run the training on self.model using the data in the train_loader,
        using the given optimizer for the given number of epochs.
        print the results.


        Parameters
        ---------

        train_loader: torch.utils.data.DataLoader
            The dataloader containing the training dataset.

        optimizer: torch.optimizer
            The optimizer to use for the training.

        epoch: int
            The number of epochs to run the training for.
        """
        batches_per_iter = len(train_loader) if not batches_per_iter else batches_per_iter
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

            if batch_idx > batches_per_iter:
                break

    def test(self, test_loader):
        """
        Run the test on self.model using the data in the test_loader and
        print the results.


        Parameters
        ---------

        test_loader: torch.utils.data.DataLoader
            The dataloader containing the test dataset.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def default_dataset(self, is_train):
        """
        Returns the default training or testing dataset.

        Parameters
        ----------

        is_train: bool
            If true return the training dataset otherwise the test.

        Returns
        -------

        torch.utils.data.Dataset
            The default train or test dataset
        """
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(
            '../data',
            train=is_train,
            download=True,
            transform=data_transform)
        return train_dataset

    def create_data_loader(self, dataset, kwargs):
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True, **kwargs
        )

    def run_train_test_loop(self, train_loader=None, test_loader=None, batches_per_iter=None):
        """
        Runs the training and test loop.

        Parameters
        ----------

        args: MNISTNetArgs
            Arguments for the train and test loop.

        train_dataset: torch.utils.data.Dataset (default None)
            The training dataset. The default full train dataset is used
            train_dataset is None.

        test_dataset: torch.utils.data.Dataset (default None)
            The testing dataset. The default full test dataset is used
            train_dataset is None.

        batches_per_iter: int (default None)
            Number of batches to train per epoch.
        """
        print("Starting training with paramaters")
        self.args.print()
        print(f"Using cuda? {self.use_cuda}")

        torch.manual_seed(self.args.seed)

        # create the training tools
        optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.args.gamma)

        # run the train-test loop
        for epoch in range(1, self.args.epochs + 1):
            # print(len(train_dataset))
            self.train(train_loader, optimizer, epoch, batches_per_iter)
            self.test(test_loader)
            scheduler.step()

        if self.args.save_model:
            torch.save(self.model.state_dict(), "mnist_cnn.pt")


def update_global_params(global_model, local_models):
    """
    Performs federated update of the global model from
    the local model according to the FedAvg algorithm.

    Parameters
    ----------

    global_model: MNISTNet
        The global model

    local_models: list of MNISTNet
        The set of local models
    """
    lm_state_dicts = [lm.state_dict() for lm in local_models]

    def agg_params(key, state_dicts):
        agg_val = state_dicts[0][key]

        for sd in state_dicts[1:]:
            agg_val = agg_val + sd[key]

        agg_val = agg_val / len(state_dicts)

        return torch.tensor(agg_val.numpy())

    global_model_dict = OrderedDict()

    for key in lm_state_dicts[0].keys():
        global_model_dict[key] = agg_params(key, lm_state_dicts)

    global_model.load_state_dict(global_model_dict)



def update_local_params(global_model, local_models):
    """
    Performs federated update of the local model from
    the global model according to the FedAvg algorithm.

    Parameters
    ----------

    global_model: MNISTNet
        The global model

    local_models: list of MNISTNet
        The set of local models
    """
    global_state_dict = global_model.state_dict()

    for lm in local_models:
        lm.load_state_dict(global_state_dict)


def fed_avg():

    digit_classes = [[0, 1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9],
                     list(range(0, 10))]

    local_idxs = [0, 1, 2]

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    args = MNISTNetArgs()
    args.epochs = 1

    mnist_train_ds = datasets.MNIST('../data', train=True, download=True, transform=data_transform)
    mnist_test_ds = datasets.MNIST('../data', train=False, download=True, transform=data_transform)

    # create the datasets
    train_loaders = [MNISTSubSet(mnist_train_ds, digit_classes[i], args=args, transform=data_transform).get_data_loader()
                      for i in range(len(digit_classes))]

    test_loaders = [MNISTSubSet(mnist_test_ds, digit_classes[i], args=args, transform=data_transform).get_data_loader()
                     for i in range(len(digit_classes))]

    steps_per_iter = 10

    # model = MNISTNet()
    # mnist_model_trainer = MNISTModelTrainer(args, model)
    # mnist_model_trainer.run_train_test_loop(
    #     train_datasets[0].get_data_loader(),
    #     test_datasets[0].get_data_loader())

    # initialize the local model trainers
    model_trainers = []
    local_models = []
    for i in local_idxs:
        local_model = MNISTNet()
        local_models.append(local_model)
        mnist_model_trainer = MNISTModelTrainer(args, local_model)
        model_trainers.append(mnist_model_trainer)

    # initialize the global model trainer
    global_model = MNISTNet()

    batches_per_iter = 10
    num_iter = 10
    for iter in range(num_iter):
        # train the local models
        for i in range(len(model_trainers)):
            model_trainers[i].run_train_test_loop(
                train_loaders[i],
                test_loaders[i],
                batches_per_iter)

        print("Performing federated updates...")
        update_global_params(global_model, local_models)
        update_local_params(global_model, local_models)
        print("Done.")

    global_trainer = MNISTModelTrainer(args, global_model)
    global_trainer.test(test_loaders[-1])



if __name__ == '__main__':

    fed_avg()
    # args = MNISTNetArgs()
    # args.epochs = 1
    #
    # digit_classes = [list(range(0, 10)),
    #                  [0, 1, 2, 3],
    #                  [4, 5, 6],
    #                  [7, 8, 9]]
    #
    # data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    #
    # mnist_train_ds = datasets.MNIST('../data', train=True, download=True, transform=data_transform)
    # mnist_test_ds = datasets.MNIST('../data', train=False, download=True, transform=data_transform)
    #
    # # create the datasets
    # train_datasets = [MNISTSubSet(mnist_train_ds, digit_classes[i], args=args, transform=data_transform)
    #                   for i in range(len(digit_classes))]
    #
    # test_datasets = [MNISTSubSet(mnist_test_ds, digit_classes[i], args=args, transform=data_transform)
    #                  for i in range(len(digit_classes))]
    #
    # model = MNISTNet()
    # mnist_model_trainer = MNISTModelTrainer(args, model)
    # mnist_model_trainer.run_train_test_loop(
    #     train_datasets[0].get_data_loader(),
    #     test_datasets[0].get_data_loader())
