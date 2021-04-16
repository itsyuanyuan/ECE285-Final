import torch 
import numpy as np 
import random
import torchvision.transforms as transforms
def set_random_seed(seed = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sample_from_noise(batch_size, noise_channel):
    return torch.randn(size = (batch_size, noise_channel,1,1))

def transform():
    d_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    return d_transforms
