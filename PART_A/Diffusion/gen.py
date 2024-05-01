import torch
import numpy as np
import matplotlib.pyplot as plt
from model import DiffusionModel

model = DiffusionModel.load_from_checkpoint("diffusion_model.pth")
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

num_images = 36 

noise = torch.randn(num_images, 1, 28, 28).to(device)

with torch.no_grad():
    for t in range(model.num_timesteps, 0, -1):
        t_batch = torch.full((num_images,), t - 1, dtype=torch.long).to(device)
        alpha = torch.cos(model.betas[t]).pow(2).to(device)
        alpha_bar = torch.sin(model.betas[t]).pow(2).to(device)
        noise_pred = model(noise, t_batch)
        noise = (
            noise - (1 - alpha) / (torch.sqrt(1 - alpha_bar)) * noise_pred
        ) / torch.sqrt(alpha)
        if t > 1:
            noise += torch.randn_like(noise) * torch.sqrt(alpha_bar)

generated_images = noise.cpu().numpy()

fig, axs = plt.subplots(6, 6, figsize=(15, 15))
for i in range(6):
    for j in range(6):
        axs[i, j].imshow(generated_images[i * 6 + j, 0], cmap="gray")
        axs[i, j].axis("off")
plt.show()
