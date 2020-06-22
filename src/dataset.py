import os
import numpy as np
import torch
import gzip
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from sklearn import preprocessing

root = os.path.dirname(os.getcwd())
colab_root = '/content/drive/My Drive/PPDAE'
exalearn_root = '/home/jorgemarpa/data/PPD'


# load pkl synthetic light-curve files to numpy array
class ProtoPlanetaryDisks(Dataset):
    def __init__(self, machine='local',
                 subsample=False, pp_scaler='MinMaxScaler'):
        if machine == 'local':
            ppd_path = '%s/data/PPD' % (root)
        elif machine == 'colab':
            ppd_path = '%s/' % (colab_root)
        elif machine == 'exalearn':
            ppd_path = '%s/' % (exalearn_root)
        else:
            print('Wrong host, please select local, colab or exalearn')
            sys.exit()

        self.meta = np.load('%s/param_arr.npy' % (ppd_path))
        self.meta = self.meta.astype(np.float32)
        self.meta_names = ['m_dust', 'Rc', 'f_exp', 'H0', 
                           'Rin', 'sd_exp', 'a_max', 'inc']
        self.imgs = np.load('%s/param_arr.npy' % (ppd_path))
        self.imgs = self.imgs.astype(np.float32)
        del self.aux
        if subsample:
            idx = np.random.randint(0, self.meta.shape[0], 1000)
            self.imgs = self.imgs[idx]
            self.meta = self.meta[idx]
        
        # if scaler == 'MinMaxScaler':
        #     self._scaler = preprocessing.MinMaxScaler()
        # if scaler == 'Normalizer':
        #     self._scaler = preprocessing.Normalizer()
        # self._scaler.fit(self.meta)
        # self.meta_p = self._scaler.transform(self.meta)

    def __getitem__(self, index):
        imgs = self.imgs[index]
        meta = self.meta[index]
        return lc, meta_p

    def __len__(self):
        return len(self.imgs)

    def get_dataloader(self, batch_size=32, shuffle=True,
                       test_split=0.2, random_seed=42):

        np.random.seed(random_seed)
        if test_split == 0.:
            train_loader = DataLoader(self, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)
            test_loader = None
        else:
            # Creating data indices for training and test splits:
            dataset_size = len(self)
            indices = list(range(dataset_size))
            split = int(np.floor(test_split * dataset_size))
            np.random.shuffle(indices)
            train_indices, test_indices = indices[split:], indices[:split]
            del indices, split

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            train_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=train_sampler,
                                      drop_last=False)
            test_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=test_sampler,
                                     drop_last=False)

        return train_loader, test_loader


class MNIST(Dataset):
    def __init__(self, machine='local'):
        if machine == 'local':
            mnist_path = '%s/data/' % (root)
        elif machine == 'colab':
            mnist_path = '%s/' % (colab_root)
        elif machine == 'exalearn':
            mnist_path = '%s/' % (exalearn_root)
        self.train = torchvision.datasets.MNIST(
            mnist_path, train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.test = torchvision.datasets.MNIST(
            mnist_path, train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))

    def __getitem__(self, index):
        return self.test[index]

    def __len__(self):
        return len(self.train) + len(self.test)

    def get_dataloader(self, batch_size=32, shuffle=True,
                       test_split=.2, random_seed=32):

        train_loader = torch.utils.data.DataLoader(self.train,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle)

        test_loader = torch.utils.data.DataLoader(self.test,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle)

        return train_loader, test_loader
