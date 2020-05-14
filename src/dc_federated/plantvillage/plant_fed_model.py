"""
Contains PlantVillage dataset specfic extension of FedAvgModelTrainer in
MobileNetv2Trainer + associated helper class.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import yaml
import csv
import os
from PIL import Image


from dc_federated.fed_avg.fed_avg_model_trainer import FedAvgModelTrainer


class MobileNetV2Args(object):
    """
    Class to abstract the arguments for a MobileNetv2ModelTrainer .
    """
    def __init__(self):
        cfg = open("PlantVillage_cfg.yaml", 'r')
        cfg_dict = yaml.load(cfg)

        self.batch_size = cfg_dict['batch_size']
        self.test_batch_size = cfg_dict['test_batch_size']
        self.epochs = cfg_dict['epochs']
        self.lr = cfg_dict['lr']
        self.gamma = cfg_dict['gamma']
        self.seed = cfg_dict['seed']
        self.log_interval = cfg_dict['log_interval']
        self.save_model = cfg_dict['save_model']
        self.training_stats_path = cfg_dict['training_stats_path']

    def print(self):
        print(f"batch_size: {self.batch_size}")
        print(f"test_batch_size: {self.test_batch_size}")
        print(f"epochs: {self.epochs}")
        print(f"lr: {self.lr}")
        print(f"gamma: {self.gamma}")
        print(f"seed: {self.seed}")
        print(f"log_interval: {self.log_interval}")
        print(f"save_model: {self.save_model}")


class PlantVillageSubSet(torch.utils.data.Dataset):
    """
    Represents a PlantVillage dataset subset. This class wraps around the
    ImageFolder class object to deliver only a subset of the PlantVillage
    dataset in the train and test sets.

    Parameters
    ----------

    plant_ds: dataset.ImageFolder dataset
        The PlantVillage dataset object based on ImageFolder module.

    args: MobileNetv2Args (default None)
        The arguments for the training.

    transform: torch.transforms (default None)
        The transformation to apply to the inputs
    """
    def __init__(self, plant_ds,  args=None, transform=None):

        self.args = MobileNetV2Args() if not args else args
        self.data = plant_ds
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


class MobileNetV2Trainer(FedAvgModelTrainer):
    """
    Trainer for the MobileNetV2 model for PlantVillage for the FedAvg algorithm. Extends
    the FedAvgModelTrainer, and implements all the relevant domain specific logic.

    Parameters
    ----------

    args: MobileNetv2Args (default None)
        The arguments for the MobileNetv2

    model: MobileNetv2 (default None)
        The model for training.

    train_loader: DataLoader (default None)
        The DataLoader for the train dataset.

    test_loader: DataLoader (default None)
        The DataLoader for the test dataset.

    batches_per_iter: int (default 10)
        The number of batches to use per train() call.

    num_classes: int (default 9)
        The number of classes to be predicted by the model.
    """
    def __init__(
            self,
            args=None,
            model=None,
            train_loader=None,
            test_loader=None,
            batches_per_iter=10,
            num_classes=9,
            global_model = False,
            checkpoints = ''):

        self.args = MobileNetV2Args() if not args else args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            'pytorch/vision:v0.5.0',
            'mobilenet_v2',
            pretrained=False,
            num_classes=num_classes).to(self.device) if not model else model

        self.train_loader = \
            PlantVillageSubSet.default_dataset(True).get_loader() if not train_loader else train_loader
        self.test_loader = \
            PlantVillageSubSet.default_dataset(False).get_loader() if not test_loader else test_loader

        self.batches_per_iter = batches_per_iter

        # for housekeeping.
        self._train_max_batches = len(self.train_loader) / self.args.batch_size
        self._train_batch_count = 0
        self._train_epoch_count = 0

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=self.args.gamma)
        self.global_model = global_model
        self.checkpoints = checkpoints
        self.best_acc = 0
        self.training_stats = {}

    def train(self):
        """
        Run the training on self.model using the data in the train_loader,
        using the given optimizer for the given number of epochs and print the model
        performance on the validation set.
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
                print(f"Training Epoch: {self._train_epoch_count} "
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

    def test(self, iteration=0):
        """
        Run the test on self.model using the data in the test_loader and
        print the results.
        """
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_acc = correct/len(self.test_loader.dataset)

        if self.global_model:
            self.record_stats(test_loss, test_acc, iteration)

        if self.args.save_model and self.global_model:
            is_best = test_acc > self.best_acc
            self.best_acc = max(test_acc, self.best_acc)
            if is_best:
                self.save_model(self.checkpoints)


        print(f"\nValidation set after epoch {self._train_epoch_count}: Average loss: {test_loss:.4f}, Accuracy: {test_acc}"
              f"({100. * correct / len(self.test_loader.dataset):.0f}%)")

    def get_model(self):
        """
        Returns the model for this trainer.

        Returns
        -------

        MobileNetV2:
            The model used in this trainer.
        """
        return self.model

    def load_model(self, model_file):
        """
        Loads a model from the given model file.

        Parameters
        -----------

        model_file: io.BytesIO or similar object
            This object should contain a serialised model.
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
            The number of batches per iteration
        """
        return self.batches_per_iter

    def save_model(self, path):
        """
        Save a model parameters.

        Parameters
        -----------

        path: str
            Folder where to save the model.
        """
        torch.save(self.model, path)

    def record_stats(self, loss, accuracy, iteration):
        """
        Save stats for the global model training as .csv file.

        Parameters
        -----------

        loss: float
            The global model validation loss.
        accuracy: float
            The global model validation accuracy.
        iteration: int
            The current iteration number.
        """
        if iteration == 1:
            self.training_stats['loss'] = []
            self.training_stats['accuracy'] = []
            self.training_stats['iteration'] = []

        self.training_stats['loss'].append(loss)
        self.training_stats['accuracy'].append(accuracy)
        self.training_stats['iteration'].append(iteration)

        with open(self.args.training_stats_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.training_stats.keys())
            writer.writerows(zip(*self.training_stats.values()))


class MobileNetV2Eval():
    """
    Evaluator for the MobileNetV2 model for PlantVillage for the FedAvg algorithm.
    Implements all the relevant domain specific logic.

    Parameters
    ----------

    args: MobileNetv2Args (default None)
        The arguments for the MobileNetv2

    model: MobileNetv2 (default None)
        The model for inference.

    test_loader: DataLoader (default None)
        The DataLoader for the test dataset.

    num_classes: int (default 9)
        The number of classes to be predicted by the model.
    """
    def __init__(
            self,
            args=None,
            model=None,
            test_loader=None,
            batches_per_iter=10,
            num_classes=9,
            single_image_pred=False
            ):

        self.args = MobileNetV2Args() if not args else args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.hub.load(
            'pytorch/vision:v0.5.0',
            'mobilenet_v2',
            pretrained=False,
            num_classes=num_classes).to(self.device) if not model else torch.load(model)

        self.test_loader = \
            PlantVillageSubSet.default_dataset(False).get_loader() if not test_loader else test_loader

        self.batches_per_iter = batches_per_iter

    def test(self, iteration=0):
        """
        Run the test on self.model using the data in the test_loader and
        print the results.
        """
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_acc = correct/len(self.test_loader.dataset)

        print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}"
              f"({100. * correct / len(self.test_loader.dataset):.0f}%)")

    def predict(self, image_file, cat, class_to_idx):
        """
        Run the inference on self.model using the input image and print the results.
        """
        self.model.eval()

        with torch.no_grad():
            image = Image.open(image_file)
            transf_image = PlantVillageSubSet.default_input_transform(False, (224,224))(image)
            data = transf_image.unsqueeze_(0)
            data = data.to(self.device)
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred_cat = list(class_to_idx.keys())[list(class_to_idx.values()).index(pred)]
            cat = class_to_idx[cat]

        print(f"\nPrediction/actual category: {pred_cat} / {pred_cat}")

    def load_model(self, model_file):
        """
        Loads a model from the given model file.

        Parameters
        -----------

        model_file: io.BytesIO or similar object
            This object should contain a serialised model.
        """
        self.model = torch.load(model_file)

    def load_model_from_state_dict(self, state_dict):
        """
        Loads a model from the given model file.

        Parameters
        -----------

        state_dict: dict
            Dictionary of parameter tensors.
        """
        self.model.load_state_dict(state_dict)

    def record_stats(self, loss, accuracy, iteration):
        """
        Save stats for the global model training as .csv file.

        Parameters
        -----------

        loss: float
            The global model validation loss.
        accuracy: float
            The global model validation accuracy.
        iteration: int
            The current iteration number.
        """
        if iteration == 1:
            self.training_stats['loss'] = []
            self.training_stats['accuracy'] = []
            self.training_stats['iteration'] = []

        self.training_stats['loss'].append(loss)
        self.training_stats['accuracy'].append(accuracy)
        self.training_stats['iteration'].append(iteration)

        with open(self.args.training_stats_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.training_stats.keys())
            writer.writerows(zip(*self.training_stats.values()))
