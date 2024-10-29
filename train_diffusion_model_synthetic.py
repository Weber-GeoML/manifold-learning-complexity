import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
from accelerate.utils import set_seed
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from data import SphereDataset, TorusDataset
from models import FullyConnectedNetwork, train_diffusion_model
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsic_dim', action="store", type=int, required=True, help="Intrinsic dimension")
    parser.add_argument('--ambient_dim', action="store", type=int, required=True, help="Ambient dimension")
    parser.add_argument('--num_epochs', action="store", type=int, required=True, help='Number of epochs to train')
    parser.add_argument('--num_points', action="store", type=int, help='Number of points to train on')
    parser.add_argument('--noise', action="store", type=float, default=0.0, help='Noise of manifold')
    parser.add_argument('--batch_size', action="store", type=int, required=True, help='Batch size')
    parser.add_argument('--lr', action="store", type=float, required=True, help='Learning Rate')
    parser.add_argument('--save_path', action="store", type=str, required=True, help='Save directory')
    parser.add_argument('--seed', action="store", type=int, default=42, help='Seed')
    parser.add_argument('--num_layers', action="store", type=int, required=False, default=5, help="Number of layers")
    parser.add_argument('--width', action="store", type=int, required=False, default=2048, help="Width of network")
    args = parser.parse_args()
    set_seed(args.seed)

    # Specify a dataset
    train_dataset = SphereDataset(args.intrinsic_dim, args.ambient_dim, args.num_points, noise=args.noise)

    # Create a dataloader from the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create a model
    model = FullyConnectedNetwork(args.ambient_dim, args.ambient_dim, args.num_layers, args.width).to(device)
    model.to(device)

    # Set the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )

    # Train
    model, loss_last_epoch = train_diffusion_model(model,train_dataloader,batch_size=args.batch_size,lr=args.lr,num_epochs=args.num_epochs)
    noise_scheduler.save_pretrained(args.save_path)
    torch.save(model,os.path.join(args.save_path,"model.pt"))
    torch.save(train_dataset[0],os.path.join(args.save_path,"data.pt"))

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Experiment took {end-start:.2f} seconds")