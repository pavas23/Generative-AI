'''
Formulae:

Noising:
mean = sqrt(alpha_t) * x0 + sqrt(1 - (alpha_t)) * noise

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST

