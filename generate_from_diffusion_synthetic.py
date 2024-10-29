import argparse
import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize, Grayscale
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler
from datasets import load_dataset
import matplotlib.pyplot as plt
from scipy.linalg import svdvals
from tqdm import tqdm
from kneed import KneeLocator
from accelerate.utils import set_seed
from typing import Optional
import time
from data import SphereDataset
from models import FullyConnectedNetwork
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_from_diffusion_synthetic(model: FullyConnectedNetwork, scheduler: DDPMScheduler, x_0: torch.Tensor, num_samples: int, timestep: int, batch_size: int):
    '''
    Generates `num_samples` samples from `pipeline` around `x_0` with noise determined by `timestep`. `batch_size` is supported.
    '''
    normal_vectors = []
    tangent_vectors = []
    samples = [x_0.repeat(1,1)]
    x_0_batch = x_0.repeat(batch_size, 1)
    timesteps = torch.LongTensor([timestep])
    timesteps_cuda = timesteps.to(device)
    with torch.no_grad():
        for i in tqdm(range(0,num_samples,batch_size)):
            if i+batch_size>num_samples:
                x_0_batch = x_0.repeat(num_samples-i, 1) 
            noise = torch.randn(x_0_batch.shape)
            x_t = scheduler.add_noise(x_0_batch, noise, timesteps)
            residuals = model(x_t.reshape(x_0_batch.shape[0],-1).to(device),timesteps_cuda).cpu()
            sample = scheduler.step(residuals,timesteps,x_t).pred_original_sample
            normal_vectors.append(residuals)
            tangent_vectors.append(x_0_batch-sample)
            samples.append(sample)
    normal_vectors = torch.concatenate(normal_vectors,dim=0).numpy()
    tangent_vectors = torch.concatenate(tangent_vectors,dim=0).numpy()
    samples = torch.concatenate(samples,dim=0).numpy()
    return normal_vectors, tangent_vectors, samples

def svd_dimension(
    vectors: np.ndarray, 
    mode: Optional[str]="normal", 
    save_dir: Optional[str]="",
) -> int:
    '''
    Computes normal/tangent space dimension by finding the best rank from SVD. Provides plot of the singular value spectrum.
    Expects `vectors` to be of shape (num_samples,total_dimension)
    Can toggle `mode` to be "normal" dimension or "intrinsic" dimension.
    Returns the dimension.
    '''
    if mode!="normal" and mode!="tangent":
        raise ValueError("'mode' must be either 'normal' or 'tangent'.")
    def numerical_rank(eigenvalues,mode="normal"):
        if mode=="normal":
            eigenvalues = np.max(eigenvalues) - eigenvalues
        return int(np.square(eigenvalues).sum()/(np.max(eigenvalues)**2))
    print(vectors.shape)
    singular_values = svdvals(vectors)[1:] # The first singular value will be very large
    explained_variances = np.cumsum(np.square(singular_values),0)/np.sum(np.square(singular_values))
    second_derivative = np.diff(explained_variances,2)
    plt.figure()
    plt.plot(second_derivative)
    plt.savefig(os.path.join(save_dir,f"{mode}_second_derivative.png"))
    estimated_dimension = numerical_rank(singular_values,mode)+1
    total_dimension = vectors.shape[1]
    plt.figure()
    plt.plot(range(1,len(singular_values)+1),singular_values,c="k")
    plt.axvline(total_dimension-estimated_dimension if mode=='normal' else estimated_dimension,c="r",alpha=0.2,label=f"Intrinsic Dimension: {estimated_dimension}")
    plt.legend()
    plt.title("Singular Values Spectrum")
    plt.xlabel("Dimension")
    plt.ylabel("Singular Value")
    plt.minorticks_on()
    plt.grid(which="major",alpha=0.15)
    plt.grid(which="minor",alpha=0.05)
    plt.savefig(os.path.join(save_dir,f"{mode}_singular.png"))
    plt.figure()
    plt.plot(range(1,len(singular_values)+1),explained_variances,c="k")
    plt.axvline(total_dimension-estimated_dimension if mode=='normal' else estimated_dimension,c="r",alpha=0.2,label=f"Intrinsic Dimension: {estimated_dimension}")
    plt.legend()
    plt.title("Explained Variance Proportion")
    plt.xlabel("Dimension")
    plt.ylabel("Explained Variance Proportion")
    plt.minorticks_on()
    plt.grid(which="major",alpha=0.15)
    plt.grid(which="minor",alpha=0.05)
    plt.savefig(os.path.join(save_dir,f"{mode}_variance.png"))
    return singular_values, explained_variances, {
        "intrinsic_dimension": int(estimated_dimension),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action="store", type=str, required=True, help='Model path')
    parser.add_argument('--intrinsic_dimension', action="store", type=int, required=True, help="Intrinsic dimension")
    parser.add_argument('--ambient_dimension', action="store", type=int, required=True, help="Intrinsic dimension")
    parser.add_argument('--num_samples', action="store", type=int, required=True, help='Number of samples to generate')
    parser.add_argument('--batch_size', action="store", type=int, required=True, help='Batch size')
    parser.add_argument('--timestep', action="store", type=int, required=True, help="Number of timesteps")
    parser.add_argument('--save_path', action="store", type=str, required=True, help='Save directory')
    parser.add_argument('--seed', action="store", type=int, default=42, help='Seed')

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_path,exist_ok=True)
    
    normal_space, tangent_space, samples = generate_from_diffusion_synthetic(
        model=torch.load(os.path.join(args.model,"model.pt")).to(device),
        scheduler=DDPMScheduler.from_pretrained(args.model),
        x_0=torch.load(os.path.join(args.model,"data.pt")),
        num_samples=args.num_samples,
        timestep=args.timestep,
        batch_size=args.batch_size
    )
    np.save(os.path.join(args.save_path,"normal_vectors.npy"),normal_space)
    np.save(os.path.join(args.save_path,"tangent_vectors.npy"),tangent_space)
    # np.save(os.path.join(args.save_path,"manifold.npy"),samples)
    normal_singular, normal_variance, normal_dimensions = svd_dimension(normal_space,mode="normal",save_dir=args.save_path)
    tangent_singular, tangent_variance, tangent_dimensions = svd_dimension(tangent_space,mode="tangent",save_dir=args.save_path)
    print("Intrinsic Dimension (from Normal)",normal_dimensions)
    print("Intrinsic Dimension (from Tangent)",tangent_dimensions)
    np.save(os.path.join(args.save_path,"normal_singular.npy"),normal_singular)
    np.save(os.path.join(args.save_path,"normal_variance.npy"),normal_variance)
    np.save(os.path.join(args.save_path,"tangent_singular.npy"),tangent_singular)
    np.save(os.path.join(args.save_path,"tangent_variance.npy"),tangent_variance)
    with open(os.path.join(args.save_path,"normal_dimensions.json"),"w") as f:
        json.dump(normal_dimensions,f)
    with open(os.path.join(args.save_path,"tangent_dimensions.json"),"w") as f:
        json.dump(tangent_dimensions,f)

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Experiment took {end-start:.2f} seconds.")