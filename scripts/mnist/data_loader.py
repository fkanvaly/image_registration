from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from itertools import repeat
import torch
import torchvision
import numpy as np


######################################
#           DATA UTILS               #
######################################

class RegisterDataset(Dataset):
    """
    Generate a selected digit data from MNIST
    """

    def __init__(self, X):
        self.X = X
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        return self.transform(self.X[idx])

    def __len__(self):
        return self.X.shape[0]


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


class BrainData:
    def __init__(self):
        npz = np.load('input/tutorial_data.npz')
        self.x_train = RegisterDataset(npz['train'])
        self.x_val = RegisterDataset(npz['validate'])
        
    def train_val(self, batch=32):
        fix_train_loader = repeater(torch.utils.data.DataLoader(self.x_train, batch_size=batch, shuffle=True))
        moving_train_loader = repeater(torch.utils.data.DataLoader(self.x_train, batch_size=batch, shuffle=True))
        
        fix_val_loader = repeater(torch.utils.data.DataLoader(self.x_val, batch_size=batch, shuffle=True))
        moving_val_loader = repeater(torch.utils.data.DataLoader(self.x_val, batch_size=batch, shuffle=True))
        
        return {"fix": fix_train_loader, "moving": moving_train_loader}, {"fix": fix_val_loader,
                                                                      "moving": moving_val_loader}
    
    def test_data(self, dataset=False):
        if dataset:
            return {"fix": self.x_val, "moving": self.x_val}

        fix_loader = repeater(torch.utils.data.DataLoader(self.x_val, batch_size=batch, shuffle=True))
        moving_loader = repeater(torch.utils.data.DataLoader(self.x_val, batch_size=batch, shuffle=True))
        
        return {"fix": fix_loader, "moving": moving_loader}
    
            
class MNISTData:
    """
    Permet de generer un dataloader pour l'entrainnement avec les chiffres qu'on veut
    Utilisation :
    >> mnist_data = MNISTData()
    >> data = mnist_data.train_val(fix_digit=2, moving_digit=3)
    """

    def __init__(self):
        (x_train_load, y_train_load), (x_test_load, y_test_load) = mnist.load_data()

        # pad images : 28x28 -> 32x32
        pad_amount = ((0, 0), (2, 2), (2, 2))

        # fix data
        self.x_train_load = np.pad(x_train_load, pad_amount, 'constant')
        self.x_test_load = np.pad(x_test_load, pad_amount, 'constant')
        self.y_train_load = y_train_load
        self.y_test_load = y_test_load

    def train_val(self, fix_digit, moving_digit, batch=32):
        """Crée deux dataloader (moving et fix) à la fois pour le train et le val"""
        # Index
        idx_digit_train = np.where((self.y_train_load == fix_digit) | (self.y_train_load == moving_digit))

        # Filter selected digit
        x_train = self.x_train_load[idx_digit_train]
        y_train = self.y_train_load[idx_digit_train]

        # train validation split
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                          stratify=y_train, random_state=42)

        # Pytorch Dataset
        # train
        fix_train_dataset = RegisterDataset(x_train[y_train == fix_digit, ...])
        moving_train_dataset = RegisterDataset(x_train[y_train == moving_digit, ...])
        # validation
        fix_val_dataset = RegisterDataset(x_val[y_val == fix_digit, ...])
        moving_val_dataset = RegisterDataset(x_val[y_val == moving_digit, ...])

        # Dataloader from dataset
        # train
        fix_train_loader = repeater(torch.utils.data.DataLoader(fix_train_dataset, batch_size=batch, shuffle=True))
        moving_train_loader = repeater(
            torch.utils.data.DataLoader(moving_train_dataset, batch_size=batch, shuffle=True))
        # val
        fix_val_loader = repeater(torch.utils.data.DataLoader(fix_val_dataset, batch_size=10, shuffle=True))
        moving_val_loader = repeater(torch.utils.data.DataLoader(moving_val_dataset, batch_size=10, shuffle=True))

        return {"fix": fix_train_loader, "moving": moving_train_loader}, {"fix": fix_val_loader,
                                                                          "moving": moving_val_loader}

    def test_data(self, fix, moving, dataset=False):
        fix_digit = self.x_test_load[self.y_test_load == fix, ...]
        moving_digit = self.x_test_load[self.y_test_load == moving, ...]
        # same moving and fix
        fix_dataset = RegisterDataset(fix_digit)
        moving_dataset = RegisterDataset(moving_digit)

        if dataset:
            return {"fix": fix_dataset, "moving": moving_dataset}

        fix_data = repeater(torch.utils.data.DataLoader(fix_dataset,
                                                        batch_size=10, shuffle=True))
        moving_data = repeater(torch.utils.data.DataLoader(moving_dataset,
                                                           batch_size=10, shuffle=True))
        return {"fix": fix_data, "moving": moving_data}

