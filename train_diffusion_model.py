import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from datasets import load_dataset
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
from accelerate.utils import set_seed
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action="store", type=str, required=True, help='Dataset Name')
    parser.add_argument('--image_size', action="store", type=int, required=True, help="Desired size of image")
    parser.add_argument('--image_channels', action="store", type=int, required=True, help="Image channels (1 for grayscale, 3 for RGB)")
    parser.add_argument('--num_epochs', action="store", type=int, required=True, help='Number of epochs to train')
    parser.add_argument('--num_points', action="store", type=int, help='Number of points to train on')
    parser.add_argument('--batch_size', action="store", type=int, required=True, help='Batch size')
    parser.add_argument('--lr', action="store", type=float, required=True, help='Learning Rate')
    parser.add_argument('--save_path', action="store", type=str, required=True, help='Save directory')
    parser.add_argument('--seed', action="store", type=int, default=42, help='Seed')

    args = parser.parse_args()
    set_seed(args.seed)


    # Specify a dataset
    train_dataset = load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
    if args.num_points:
        train_dataset = train_dataset.select(range(args.num_points))

    preprocess = Compose([
        Resize((args.image_size, args.image_size)),
        ToImage(),
        ToDtype(torch.float32,scale=True),
        Normalize([0.5], [0.5]),
    ])

    def transform(examples):
        images = [preprocess(image) for image in (examples["image"] if "image" in examples else examples["img"])]
        return {"images": images}

    train_dataset.set_transform(transform)

    # Create a dataloader from the dataset to serve up the transformed images in batches
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create a model
    model = UNet2DModel(
        sample_size=args.image_size,  # the target image resolution
        in_channels=args.image_channels,  # the number of input channels, 3 for RGB images
        out_channels=args.image_channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 256),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D"
        ),
    )
    model.to(device)

    # Set the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    losses = []

    for epoch in range(args.num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader),desc=f'Epoch {epoch}',total=len(train_dataloader)):
            clean_images = batch["images"].to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate the loss
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)
            losses.append(loss.item())

            # Update the model parameters with the optimizer
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 1 == 0:
            loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")
    
    noise_scheduler.save_pretrained(args.save_path)
    model.save_pretrained(args.save_path)

    pipeline = DDPMPipeline(unet=model,scheduler=noise_scheduler)
    example_generations = pipeline(16).images
    for i,image in enumerate(example_generations):
        image.save(os.path.join(args.save_path,f"sample_{i}.png"))

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Experiment took {end-start:.2f} seconds")