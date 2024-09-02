# Image Generation and Image-to-Image Translation

## Part A - Image Generation
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

## Part B - Image-to-Image Translation
We focus on image-to-image translation, aiming to convert source images into target images while altering specific visual properties while preserving others. This is accomplished by employing two variants of Generative Adversarial Networks (GANs) on the CelebA dataset.
- [Deep Convolutional GAN (DCGAN)](https://arxiv.org/pdf/1511.06434)
- [CycleGAN](https://arxiv.org/pdf/1703.10593)

### CycleGAN
We implement a CycleGAN (Cycle-Consistent Generative Adversarial Network) for image-to-image translation. CycleGANs are used to learn mappings between two different domains without requiring paired data.

The activation functions used are
- Leaky ReLU: Used in both the generator and discriminator.
- ReLU: Used in the generator for non-linearity.
- Sigmoid: Used in the discriminator to output probabilities.
  
We also chose to use the `Adam Optimizer`, since it generally requires a lower learning rate, and that it converges faster.

#### Adversarial Loss (GAN Loss)
Binary Cross-Entropy (BCE) loss is used to train the generators and discriminators by minimizing the difference between real and fake predictions.

#### Cycle Consistency Loss
Mean Absolute Error (L1 loss) is used to enforce cycle consistency between the original and reconstructed images, ensuring that the image after translation and back-translation is close to the original image.


### Men without glasses to men with glasses and vice versa

<img width="899" alt="1" src="https://github.com/user-attachments/assets/fd08c83f-14ff-466e-8cc9-5a8f3d0c1a8f">
<img width="897" alt="2" src="https://github.com/user-attachments/assets/5ac16fa4-9735-47f8-a6a7-b74bf434f383">
<img width="899" alt="3" src="https://github.com/user-attachments/assets/29fe6578-5c73-40c9-bccf-90bc4c2d0ce4">
<img width="899" alt="4" src="https://github.com/user-attachments/assets/a7f92979-bfe3-4e7b-aabb-978b75dc46b7">
<img width="899" alt="5" src="https://github.com/user-attachments/assets/24f439cb-5987-4b96-8590-0e7266ef332d">


### Men with glasses to women with glasses and vice versa

<img width="899" alt="1" src="https://github.com/user-attachments/assets/b3dc3097-ede3-46a7-b3a4-634ee51e0bbd">
<img width="897" alt="2" src="https://github.com/user-attachments/assets/a73b8fbd-e17a-4cb8-8d64-fce134b6de26">
<img width="899" alt="3" src="https://github.com/user-attachments/assets/19a55d4c-5b2a-4b89-bc03-9990e8b55ff6">
<img width="898" alt="4" src="https://github.com/user-attachments/assets/434a3f0c-a10a-448a-a416-e2b156ceba2d">
<img width="899" alt="5" src="https://github.com/user-attachments/assets/301ae2ee-e938-4b01-b672-bbb86f30794a">

### Deep Convolutional GAN (DCGAN)
We implement three neural network models: a Generative Adversarial Network (GAN) Generator, a GAN Discriminator, and a CNN Encoder.

The activation functions used are
- Leaky ReLU: Used in both the discriminator and encoder models
- ReLU: Used in the generator for non-linearity.
- Sigmoid: Used in the final layer of discriminator to output probabilities

The Loss function used for training is `Binary Cross-Entropy Loss`.

## Vector Arithmetic Result

### Men without glasses + People with glasses - People without glasses

<img width="733" alt="men_people" src="https://github.com/user-attachments/assets/d69ed1a1-4b68-40b7-a03e-5936c99b4f29">

### Men with glasses - Men without glasses + Women without glasses

<img width="737" alt="men_women" src="https://github.com/user-attachments/assets/a9a42945-2289-4ba9-94e3-b3665f521c46">

### Smiling Men + People with Hat - People with Hat + People with Mustache - People without Mustache

<img width="738" alt="men_hat" src="https://github.com/user-attachments/assets/678161e7-4a79-4916-8b85-cf0bfe90efe8">



