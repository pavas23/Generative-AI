import matplotlib.pyplot as plt

latent_sizes = [2, 4, 8, 16, 32, 64]

training_losses = {}
mses = {}

with open('out.log', 'r') as logfile:
    lines = logfile.readlines()
    latent_size = None
    for line in lines:
        if line.startswith("Training with latent size"):
            latent_size = int(line.split(":")[-1].strip())
            training_losses[latent_size] = []
        elif line.startswith("MSE for latent_size"):
            mses[latent_size] = float(line.split(":")[-1].strip())
        elif line.startswith("Epoch"):
            loss = float(line.split(":")[-1].strip())
            training_losses[latent_size].append(loss)

for latent_size, losses in training_losses.items():
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses)
    plt.title(f"Training Losses for Latent Size {latent_size}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"training_losses_latent_size_{latent_size}.png")
    plt.show()

plt.figure(figsize=(10, 6))
plt.plot(list(mses.keys()), list(mses.values()), marker="o", linestyle="-")
plt.title("MSE Losses for Different Latent Sizes")
plt.xlabel("Latent Size")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("mse_losses.png")
plt.show()
