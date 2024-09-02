# Image Generation and Image-to-Image Translation

## Part A
In this we train generative models to generate images from random vectors sampled from the latent space, employing Variational Autoencoders (VAEs), Generative Adversarial Networks
(GANs), and Diffusion Models on the MNIST dataset.

### VAE
The VAE is structured only with `Linear Layers`, since the simplicity of the dataset meant we weren’t required to make the model more complex with ideas like Conv and Pool Layers. Our
architecture involves the model learning the `mean` and `variance` of a latent variable using an Encoder, and re-constructing the images from that using a Decoder. We used the `relu` and `sigmoid` activation functions in the layers.

### GAN
The GAN is trained to generate realistic `MNIST` digit images. The generator learns to generate images from random noise, while the discriminator learns to distinguish between `real and fake` images. The training loop alternates between training the `discriminator and the generator`. The generator aims to minimize the discriminator's ability to distinguish fake images from real ones. The discriminator aims to maximize its ability to distinguish between real and fake images. We used the `relu` in the layers of the generator and `relu and sigmoid` activation functions in the layers of the discriminator. We also chose to use the `Adam Optimizer`, since it generally requires a lower learning rate, and that it converges faster.

The following are the images of the reconstructions of random points for latent sizes 2, 4, 8 and 64

![fake_2_50](https://github.com/user-attachments/assets/d2fa1ce0-9c87-4da1-a53e-c23527af0c11) ![fake_4_50](https://github.com/user-attachments/assets/5156f42e-46b4-43db-bd05-37f0ad4abe3a) ![fake_8_50](https://github.com/user-attachments/assets/63e821df-8f13-4fca-a8a7-bf942df75551) ![fake_64_50](https://github.com/user-attachments/assets/2504ece8-4879-487c-95b5-fc06a7370774)

The following images also depict how the training loss reduces over epochs for each latent size

### Generator
<img width="600" alt="Generator" src="https://github.com/user-attachments/assets/422f0ee7-75f7-4910-b8bd-146fced99c3c">

### Discriminator
<img width="600" alt="Discriminator" src="https://github.com/user-attachments/assets/6e7f0476-59da-4919-ba82-b3787ed20228">




