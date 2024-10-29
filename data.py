import numpy as np
import torch
from torch.utils.data import Dataset

class SphereDataset(Dataset):
    """
    Generates a random unit hypersphere with specified intrinsic_dim and ambient_dim
    """
    def __init__(self, intrinsic_dim, ambient_dim, n_points, noise=0):
        self.intrinsic_dim = intrinsic_dim
        self.ambient_dim = ambient_dim
        self.n_points = n_points
        self.noise = noise
        self.basis = np.linalg.qr(np.random.randn(ambient_dim, ambient_dim))[0][:intrinsic_dim+1]
        self.vectors = np.random.randn(n_points, self.intrinsic_dim+1)
        self.vectors = self.vectors / np.linalg.norm(self.vectors, axis = -1, keepdims=True)
        self.points = torch.tensor(self.vectors@self.basis, dtype=torch.float32)
        self.points += noise*np.random.randn(n_points, self.ambient_dim)/np.sqrt(self.ambient_dim)
        self.points = self.points.float()

    def __len__(self):
        return self.n_points

    def __getitem__(self, idx):
        return self.points[idx]