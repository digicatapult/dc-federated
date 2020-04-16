"""
Contains PlantVillage dataset specfic extension of FedAvgModelTrainer in
MobileNetv2Trainer + associtaed helper class.
"""

import numpy as np

import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import torch.utils.model_zoo as model_zoo

from dc_federated.fed_avg.fed_avg_model_trainer import FedAvgModelTrainer


class MobileNetv2(nn.Module):
    """
    MobileNetv2 from torchvision.
    """
    def __init__(self, num_classes):
        super(MobileNetv2, self).__init__()

        model_urls = {
            'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
        }

        # Obtain the desired model from the pretrainedmodels library
        self.model = models.__dict__['mobilenet_v2'](num_classes=num_classes)

        # Loads the Torch serialized pretrained model at the given URL.
        pretrained_dict = model_zoo.load_url(model_urls['mobilenet_v2'])

        # Filter out unnecessary keys in pretraine model state dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model.state_dict()}
        # Overwrite entries in the model state dict
        self.model.state_dict().update(pretrained_dict)

        # Load the new state dict
        self.model.load_state_dict(self.model.state_dict())


    def forward(self, x):
        return self.model(x)


class MobileNetv2Args(object):
    """
    Class to abstract the arguments for a MobileNetv2ModelTrainer .
    """
    def __init__(self):
        self.batch_size = 16
        self.test_batch_size = 16
        self.epochs = 20
        self.lr = 0.001
        self.gamma = 0.7
        # self.no_cuda = False
        self.seed = 1
        self.log_interval = 5
        self.save_model = False


    def print(self):
        print(f"batch_size: {self.batch_size}")
        print(f"test_batch_size: {self.test_batch_size}")
        print(f"epochs: {self.epochs}")
        print(f"lr: {self.lr}")
        print(f"gamma: {self.gamma}")
        # print(f"no_cuda: {self.no_cuda}")
        print(f"seed: {self.seed}")
        print(f"log_interval: {self.log_interval}")
        print(f"save_model: {self.save_model}")


class PlantVillageSubSet(torch.utils.data.Dataset):
    """
    Represents a PlantVillage dataset subset. This class deliver only a subset of
    digits in the train and test sets.

    Parameters
    ----------

    plant_ds: dataset.ImageFolder dataset
        The PlantVillage dataset object based on ImageFolder module.

    args: MobileNetv2Args (default None)
        The arguments for the training.

    input_transform: torch.transforms (default None)
        The transformation to apply to the inputs

    target_transform: torch.transforms (default None)
        The transformation to apply to the target
    """
    def __init__(self, plant_ds,  args=None, transform=None):

        # self.digits = digits
        self.args = MobileNetv2Args() if not args else args
        # mask = np.isin(mnist_ds.targets, digits)
        # self.data = mnist_ds.data[mask].clone()
        # self.targets = mnist_ds.targets[mask].clone()
        # self.input_transform = input_transform
        self.data = plant_ds
        # self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        """
        Implementation of a function required to extend Dataset. Returns\
        the dataset item at the given index.

        Parameters
        ----------
        index: int
            The index of the input + target to return.

        Returns
        -------

        tuple of a pair of torch.tensors
            The input and the target.
        """

        return self.data.__getitem__(index)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------

        int:
            The length of the dataset.
        """
        return len(self.data)

    def get_data_loader(self, use_cuda=True, kwargs=None):
        """
        Returns a DataLoader object for this dataset


        Parameters
        ----------

        use_cuda: bool (default False)
            Whether to use cuda or not.

        kwargs: dict
            Paramters to pass on to the DataLoader


        Returns
        -------

        DataLoader:
            A dataloader for this dataset.
        """

        if kwargs is None:
            kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        return torch.utils.data.DataLoader(
            self, batch_size=self.args.batch_size, shuffle=True, **kwargs)

    @staticmethod
    def default_input_transform(is_train, resize):
        """
        Returns the input transformation for train, valid and test.
        Parameters
        ---------
        is_train: bool
            Wether to use specific transformations for train set

        resize: tuple of integers
            input sizes for the model used

        Returns
        -------

        torch.transforms:
            A default set of transformations for the inputs.
        """
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomSizedCrop(max(resize)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(max(resize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        return data_transforms['train'] if is_train else data_transforms['val']

    @staticmethod
    def default_plant_ds(root = "../data", transform=None):
        """
        Returns a default dataset.ImageFolder dataset.

        Parameters
        ---------

        transform: transforms
            The input data transformation to use

        Returns
        -------

        dataset.Dataset:
            The default plant dataset.
        """
        print('data_path', root)
        return datasets.ImageFolder(root = root, transform=transform)

    @staticmethod
    def default_dataset(is_train, root, resize):
        """
        Returns a train or test dataset on the whole dataset.

        Parameters
        ----------

        is_train: bool
            Whether to return the training or test dataset.

        Returns
        -------

        PlantVillageSubSet:
            The train or test dataset.
        """
        data_transform = PlantVillageSubSet.default_input_transform(is_train, resize)

        return PlantVillageSubSet(
            PlantVillageSubSet.default_plant_ds(root, data_transform),
            transform=data_transform
        )


class MobileNetv2Trainer(FedAvgModelTrainer):
    """
    Trainer for the Mobilenetv2 model for the FedAvg algorithm. Extends
    the FedAvgModelTrainer, and implements all the relevant domain
    specific logic.

    Parameters
    ----------

    args: MobileNetv2Args (default None)
        The arguments for the MobileNetv2

    model: MobileNetv2 (default None)
        The model for training.

    train_loader: DataLoader
        The DataLoader for the train dataset.

    test_loader: DataLoader
        The DataLoader for the test dataset.

    batches_per_iter: int
        The number of batches to use per train() call.
    """
    def __init__(
            self,
            args=None,
            model=None,
            train_loader=None,
            test_loader=None,
            batches_per_iter=10,
            num_classes=9):

        self.args = MobileNetv2Args() if not args else args

        self.use_cuda = True #not self.args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = MobileNetv2(num_classes).to(self.device) if not model else model

        self.train_loader = \
            PlantVillageSubSet.default_dataset(True).get_loader() if not train_loader else train_loader
        self.test_loader = \
            PlantVillageSubSet.default_dataset(False).get_loader() if not test_loader else test_loader

        self.batches_per_iter = batches_per_iter

        # for housekeepiing.
        self._train_max_batches = len(self.train_loader) / self.args.batch_size
        self._train_batch_count = 0
        self._train_epoch_count = 0

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=self.args.gamma)

    def train(self):
        """
        Run the training on self.model using the data in the train_loader,
        using the given optimizer for the given number of epochs.
        print the results.
        """
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target).to(self.device)
            loss.backward()
            self.optimizer.step()
            if self._train_batch_count % self.args.log_interval == 0:
                print(f"Training Epoch: {self._train_epoch_count}"
                      f"Progress: [{self._train_batch_count * len(data)}/{len(self.train_loader.dataset)}"
                      f"({100. * self._train_batch_count / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            self._train_batch_count += 1
            if batch_idx > self.batches_per_iter:
                break

        # housekeeping after a single epoch
        if self._train_batch_count >= len(self.train_loader):
            self._train_epoch_count += 1
            self._train_batch_count = 0
            self.scheduler.step()


    def test(self):
        """
        Run the test on self.model using the data in the test_loader and
        print the results.
        """
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
            # for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print(f"\nValidation set after epoch {self._train_epoch_count}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)}"
              f"({100. * correct / len(self.test_loader.dataset):.0f}%)")

    def get_model(self):
        """
        Returns the model for this trainer.

        Returns
        -------

        MobileNetv2:
            The model used in this trainer.
        """
        return self.model

    def load_model(self, model_file):
        """
        Loads a model from the given model file.

        Parameters
        -----------

        model_file: io.BytesIO or similar object
            This object should contain a serilaized model.
        """
        self.model = torch.load(model_file)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=self.args.gamma)

    def load_model_from_state_dict(self, state_dict):
        """
        Loads a model from the given model file.

        Parameters
        -----------

        state_dict: dict
            Dictionary of parameter tensors.
        """
        self.model.load_state_dict(state_dict)

    def get_per_session_train_size(self):
        """
        Returns the number of batches per iter as the per session train size as
        the number of batches and size of batches are same.

        Returns
        -------

        int:
            The number of batches per ietration
        """
        return self.batches_per_iter
