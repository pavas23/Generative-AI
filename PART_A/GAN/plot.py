import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import re

with open("out.log", "r") as file:
    log_data = file.read()

d_losses = re.findall(r"d_loss: (\d+\.\d+)", log_data)
g_losses = re.findall(r"g_loss: (\d+\.\d+)", log_data)

# Convert the losses to floats
d_losses = [float(loss) for loss in d_losses]
g_losses = [float(loss) for loss in g_losses]

latent_sizes = [2, 4, 8, 16, 32, 64]
colors = cm.rainbow(np.linspace(0, 1, len(latent_sizes)))

legend_handles = []

epochs_per_segment = len(g_losses) // len(latent_sizes)
epochs = range(1, len(g_losses) + 1)

for i in range(len(latent_sizes)):
    start_idx = i * epochs_per_segment
    end_idx = (
        (i + 1) * epochs_per_segment
        if (i + 1) * epochs_per_segment < len(g_losses)
        else len(g_losses)
    )

    plt.plot(
        epochs[start_idx:end_idx],
        g_losses[start_idx:end_idx],
        color=colors[i],
    )

    legend_handles.append(
        plt.Line2D(
            [0], [0], color=colors[i], lw=2, label=f"Latent Size {latent_sizes[i]}"
        )
    )

plt.title("Training Losses")
plt.xlabel("Trained over 50 epochs for each latent size")
plt.ylabel("Loss")

custom_ticks = []
plt.xticks(custom_ticks, labels=custom_ticks)

plt.legend(handles=legend_handles)
plt.show()
