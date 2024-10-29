from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
device = "cuda" if torch.cuda.is_available() else "cpu"

class FullyConnectedNetwork(nn.Module):
    """
    A standard ReLU feed forward neural network
    """
    def __init__(self, input_dim, output_dim, num_layers, width):
        super(FullyConnectedNetwork, self).__init__()
        first_layers = []
        first_layers.append(nn.Linear(input_dim, width))
        first_layers.append(nn.ReLU())
        first_layers.append(nn.Linear(width, width))

        self.first_layer = nn.Sequential(*first_layers)

        layers = []
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))
        self.fc_layers = nn.Sequential(*layers)

        time_layers = []
        time_layers.append(nn.Linear(1, width))
        time_layers.append(nn.ReLU())
        time_layers.append(nn.Linear(width, width))
        self.time_layers = nn.Sequential(*time_layers)

    def forward(self, x, timesteps = None, return_dict=None):
        if len(x.shape)>2:
            x = x.squeeze()
        x = self.first_layer(x)
        if timesteps is not None:
            x += self.time_layers(timesteps.type(torch.float32).reshape(-1,1)/1000.)
        return self.fc_layers(x)

def train_diffusion_model(model,train_dataloader,lr=1e-4,num_epochs=50,batch_size=16):
    """
    Trains a diffusion model on the specified dataloader
    """
    # Set the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader),desc=f'Epoch {epoch}',total=len(train_dataloader)):
            clean_images = batch.to(device).reshape(batch.shape[0], 1, -1)
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
            # noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            noise_pred = model(noisy_images.reshape(bs,-1),timesteps)

            # Calculate the loss
            # loss = F.mse_loss(noise_pred, noise)
            loss = F.mse_loss(noise_pred, noise.reshape(bs,-1))
            loss.backward(loss)
            losses.append(loss.item())

            # Update the model parameters with the optimizer
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 1 == 0:
            loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")
    return model, loss_last_epoch