import torch
import matplotlib.pyplot as plt
from vae import latent_sizes, VAE, test_loader

for latent_size in latent_sizes:
    print(f"Generating Images for latent size: {latent_size}")
    vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=latent_size)
    if torch.cuda.is_available():
        vae.cuda()
    vae.load_state_dict(torch.load(f"chkpts/vae_{latent_size}.pt"))
    vae.eval()

    for classlbl in range(10):
        for data, labels in test_loader:
            indices = (labels == classlbl).nonzero()
            if len(indices) > 0:
                image = (
                    data[indices[0]].cuda()
                    if torch.cuda.is_available()
                    else data[indices[0]]
                )
                break
        with torch.inference_mode():
            mu, logvar = vae.encoder(image.view(-1, 784))
            z = vae.sampling(mu, logvar)
            recon = vae.decoder(z).cpu().detach().numpy().reshape(28, 28)

        img_pth = f"generated_imgs/vae_{latent_size}_class_{classlbl}.png"
        plt.imsave(img_pth, recon, cmap="gray")

        print(
            f"Generated Image for latent size: {latent_size}, class: {classlbl} saved at {img_pth}"
        )
