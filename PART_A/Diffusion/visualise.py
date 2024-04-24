import torch
import matplotlib.pyplot as plt

def generate_samples(model, diffusion, num_samples=16):
    samples = torch.randn(num_samples, 1, 28, 28)
    for t in reversed(range(diffusion.num_steps)):
        predicted_mean, _ = diffusion.p_mean_variance(model, samples, t)
        samples = predicted_mean + torch.randn_like(samples) * diffusion.beta[t].sqrt()

    return samples.cpu().detach()


def visualize_samples(samples, title):
    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axs.flat):
        ax.imshow(samples[i, 0], cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.savefig(f"{title}.png")
