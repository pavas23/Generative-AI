import pytorch_lightning as pl
import torch
from model import DiffusionModel

num_timesteps = 469
model = DiffusionModel(num_timesteps)

trainer = pl.Trainer(
    max_epochs=100,
    gpus=1 if torch.cuda.is_available() else None,
    enable_progress_bar=True,
)

trainer.fit(model)
torch.save(model.state_dict(), "diffusion_model.pth")
