# Image Generation and Image-to-Image Translation

## Part A
In this we train generative models to generate images from random vectors sampled from the latent space, employing Variational Autoencoders (VAEs), Generative Adversarial Networks
(GANs), and Diffusion Models on the MNIST dataset.

### VAE
The VAE is structured only with `Linear Layers`, since the simplicity of the dataset meant we weren’t required to make the model more complex with ideas like Conv and Pool Layers.

Our architecture involves the model learning the `mean` and `variance` of a latent variable using an Encoder, and re-constructing the images from that using a Decoder. We used the `relu` and `sigmoid` activation functions in the layers.

### GAN
The GAN is trained to generate realistic `MNIST` digit images. The generator learns to generate images from random noise, while the discriminator learns to distinguish between `real and fake` images. The training loop alternates between training the `discriminator and the generator`.

The generator aims to minimize the discriminator's ability to distinguish fake images from real ones. The discriminator aims to maximize its ability to distinguish between real and fake images.

We used the `relu` in the layers of the generator and `relu and sigmoid` activation functions in the layers of the discriminator. We also chose to use the `Adam Optimizer`, since it generally requires a lower learning rate, and that it converges faster.

The following are the images of the reconstructions of random points for latent sizes 2, 4 and 8.

![fake_2_50](https://github.com/user-attachments/assets/d2fa1ce0-9c87-4da1-a53e-c23527af0c11) ![fake_4_50](https://github.com/user-attachments/assets/5156f42e-46b4-43db-bd05-37f0ad4abe3a) ![fake_8_50](https://github.com/user-attachments/assets/63e821df-8f13-4fca-a8a7-bf942df75551)

The following image also depicts how the training loss reduces over epochs for each latent size for the generator.

<img width="600" alt="Generator" src="https://github.com/user-attachments/assets/422f0ee7-75f7-4910-b8bd-146fced99c3c">

### Diffusion Models
We implement a diffusion model  for the MNIST dataset. Diffusion models are used for image generation and manipulation tasks. This model applies forward and reverse diffusion processes to images, allowing for the generation of high-quality samples.

`SiLU (Sigmoid-weighted Linear Unit)` is used as the activation function.

We also chose to use the Adam Optimizer, since it generally requires a lower learning rate, and that it converges faster.

These are the results obtained from the diffusion model in multiple steps.

![steps_00000469](https://github.com/user-attachments/assets/e2918672-5c47-45aa-910f-d28f456a5ef1) ![steps_00006097](https://github.com/user-attachments/assets/19f50a9b-a6ab-4437-af0f-711c10405b1a) ![steps_00015946](https://github.com/user-attachments/assets/697a00b7-f2e0-4a47-a855-cf07ad101892)






