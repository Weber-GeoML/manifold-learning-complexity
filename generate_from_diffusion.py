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
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_from_diffusion(pipeline: DDPMPipeline, x_0: torch.Tensor, num_samples: int, timestep: int, batch_size: int):
    '''
    Generates `num_samples` samples from `pipeline` around `x_0` with noise determined by `timestep`. `batch_size` is supported.
    '''
    generator = pipeline.to(device)
    normal_vectors = []
    tangent_vectors = []
    samples = [x_0.repeat(1,1,1,1)]
    x_0_batch = x_0.repeat(batch_size, 1, 1, 1)
    with torch.no_grad():
        for i in tqdm(range(0,num_samples,batch_size)):
            if i+batch_size>num_samples:
                x_0_batch = x_0.repeat(num_samples-i, 1, 1, 1) 
            noise = torch.randn(x_0_batch.shape)
            timesteps = torch.LongTensor([timestep])
            x_t = generator.scheduler.add_noise(x_0_batch, noise, timesteps)
            residuals = generator.unet(x_t.to(device),timesteps.to(device)).sample.detach().cpu()
            sample = generator.scheduler.step(residuals,timesteps,x_t).pred_original_sample
            normal_vectors.append(residuals)
            tangent_vectors.append(x_0_batch-sample)
            samples.append(sample)
    normal_vectors = torch.concatenate(normal_vectors,dim=0).flatten(start_dim=1).numpy()
    tangent_vectors = torch.concatenate(tangent_vectors,dim=0).flatten(start_dim=1).numpy()
    samples = torch.concatenate(samples,dim=0).flatten(start_dim=1).numpy()
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
    Can also toggle `elbow_method to` determine elbow point between "max_difference", "second_derivative", or "max_curvature".
    Returns the dimension.
    '''
    if mode!="normal" and mode!="tangent":
        raise ValueError("'mode' must be either 'normal' or 'tangent'.")
    singular_values = svdvals(vectors)[1:] # The first singular value will be very large
    explained_variances = np.cumsum(np.square(singular_values),0)/np.sum(np.square(singular_values))
    second_derivative = np.diff(explained_variances,2)
    plt.figure()
    plt.plot(second_derivative)
    plt.savefig(os.path.join(save_dir,f"{mode}_second_derivative.png"))
    estimated_dimension = np.argmax(np.abs(singular_values[1:]-singular_values[:-1]))+1
    estimated_dimension2 = np.argmax(np.abs(second_derivative)[2:])+2+1
    estimated_dimension3 = round(KneeLocator(range(1,len(explained_variances)+1),explained_variances).knee)
    total_dimension = vectors.shape[1]
    plt.figure()
    plt.plot(range(1,len(singular_values)+1),singular_values,c="k")
    plt.axvline(estimated_dimension,c="r",alpha=0.2,label=f"Max Difference: {total_dimension-estimated_dimension if mode=='normal' else estimated_dimension}")
    plt.axvline(estimated_dimension2,c="b",alpha=0.2,label=f"Second Derivative: {total_dimension-estimated_dimension2 if mode=='normal' else estimated_dimension2}")
    plt.axvline(estimated_dimension3,c="g",alpha=0.2,label=f"Max Curvature: {total_dimension-estimated_dimension3 if mode=='normal' else estimated_dimension3}")
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
    plt.axvline(estimated_dimension,c="r",alpha=0.2,label=f"Max Difference: {total_dimension-estimated_dimension if mode=='normal' else estimated_dimension}")
    plt.axvline(estimated_dimension2,c="b",alpha=0.2,label=f"Second Derivative: {total_dimension-estimated_dimension2 if mode=='normal' else estimated_dimension2}")
    plt.axvline(estimated_dimension3,c="g",alpha=0.2,label=f"Max Curvature: {total_dimension-estimated_dimension3 if mode=='normal' else estimated_dimension3}")
    plt.legend()
    plt.title("Explained Variance Proportion")
    plt.xlabel("Dimension")
    plt.ylabel("Explained Variance Proportion")
    plt.minorticks_on()
    plt.grid(which="major",alpha=0.15)
    plt.grid(which="minor",alpha=0.05)
    plt.savefig(os.path.join(save_dir,f"{mode}_variance.png"))
    return singular_values, explained_variances, {
        "max_difference": int(total_dimension-estimated_dimension if mode=='normal' else estimated_dimension),
        "second_derivative": int(total_dimension-estimated_dimension2 if mode=='normal' else estimated_dimension2),
        "max_curvature": int(total_dimension-estimated_dimension3 if mode=='normal' else estimated_dimension3)
    }

def LinearTimeSVD(A, c, k, p):
    m, n = A.shape
    c = round(c)
    k = round(k)

    # Checking inputs
    if not (k >= 1 and k <= c and c <= n and np.min(p) >= 0):
        print(f'k = {k}, c = {c}, n = {n}, min(p) = {np.min(p)}')
        raise ValueError('Input error!')

    # Main iteration
    C = np.zeros((m, c))
    for t in range(c):
        it = np.random.choice(range(n),p=p)
        C[:, t] = A[:, it] / np.sqrt(c * p[it])

    return np.sqrt(svdvals(C.T@C))[:k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action="store", type=str, required=True, help='Model path')
    parser.add_argument('--image_size', action="store", type=int, required=True, help="Desired size of image")
    parser.add_argument('--dataset', action="store", type=str, required=True, help='Dataset path')
    parser.add_argument('--x_0', action="store", type=str, required=True, help='Around which point to generate')
    parser.add_argument('--timestep', action="store", type=int, default=1, help="Timestep to determine noise level")
    parser.add_argument('--num_samples', action="store", type=int, required=True, help='Number of samples to generate')
    parser.add_argument('--batch_size', action="store", type=int, required=True, help='Batch size')
    parser.add_argument('--save_path', action="store", type=str, required=True, help='Save directory')
    parser.add_argument('--seed', action="store", type=int, default=42, help='Seed')

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_path,exist_ok=True)
    
    pipeline = DDPMPipeline(UNet2DModel.from_pretrained(args.model),DDPMScheduler.from_pretrained(args.model))
    
    train_dataset = load_dataset(args.dataset, split="train")
    preprocess = Compose([
        Resize((args.image_size, args.image_size)),
        ToImage(),
        ToDtype(torch.float32,scale=True),
        Normalize([0.5], [0.5]),
    ])
    def transform(examples):
        images = [preprocess(image) for image in (examples["image"] if "image" in examples else examples["img"])]
        if "label" in examples:
            labels = [label for label in examples["label"]]
            return {"images": images, "labels": labels}
        else:
            return {"images": images}
    train_dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    if args.x_0=="random": # Get first sample
        x_0 = next(iter(train_dataloader))["images"]
    elif re.match(r"label_(.*)",args.x_0): # Interpret as first instance of specific label
        m = re.match(r"label_(.*)",args.x_0)
        for batch in train_dataloader:
            if str(batch["labels"][0].item())==m.group(1):
                x_0 = batch["images"][0]
                break
    else: # Interpret as path
        x_0 = preprocess(PIL.Image.open(args.x_0))
    img_array = ((x_0*0.5+0.5).squeeze().numpy()*255).astype(np.uint8)
    if img_array.shape[0]==3:
        img_array = img_array.transpose(1,2,0)
    PIL.Image.fromarray(img_array).save(os.path.join(args.save_path,"x_0.png"))

    normal_space, tangent_space, samples = generate_from_diffusion(
        pipeline=pipeline,
        x_0=x_0,
        num_samples=args.num_samples,
        timestep=args.timestep,
        batch_size=args.batch_size
    )
    np.save(os.path.join(args.save_path,"normal_vectors.npy"),normal_space)
    np.save(os.path.join(args.save_path,"tangent_vectors.npy"),tangent_space)
    np.save(os.path.join(args.save_path,"manifold.npy"),samples)
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