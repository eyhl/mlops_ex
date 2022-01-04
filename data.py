import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class mnistDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]
        
        if self.transform:
            x = Image.fromarray(self.images[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        #else:
        #    x = torch.from_numpy(x)
            
        # if image is greyscale add unit color channel
        # torch expects (n_samples, channels, height, width)
        if len(x.shape) <= 3:
            x = x[np.newaxis, :, :]

        return torch.from_numpy(x).float(), y
    
    def __len__(self):
        return len(self.images)


def mnist_loader(path='../../corruptmnist', n_files=5, image_scale=255):
    """
    Loads .npz corruptedmnist, assumes loaded image values to be between 0 and 1
    """
    # load and stack the corrupted mnist dataset
    train_images = np.vstack([np.load(path + '/train_{}.npz'.format(str(i)))['images'] for i in range(n_files)])
    train_labels = np.hstack([np.load(path + '/train_{}.npz'.format(str(i)))['labels'] for i in range(n_files)]) 
    
    test_images = np.load(path + '/test.npz')['images']
    test_labels = np.load(path + '/test.npz')['labels']

    return train_images * image_scale, train_labels, test_images * image_scale, test_labels

