import re
import matplotlib.pyplot as plt

def process_file(file_path):
    epoch_losses = {}
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"Epoch \[(\d+)\/\d+\] Loss: ([\d.e-]+)", line)
            if match:
                epoch_number = int(match.group(1))
                loss_value = float(match.group(2))
                if epoch_number not in epoch_losses:
                    epoch_losses[epoch_number] = []
                epoch_losses[epoch_number].append(loss_value)

    # Calculate average loss for each epoch
    epoch_average_losses = {}
    for epoch_number, losses in epoch_losses.items():
        average_loss = sum(losses) / len(losses)
        epoch_average_losses[epoch_number] = average_loss

    return epoch_average_losses

def plot_loss_vs_epoch(epoch_losses):
    epochs = list(epoch_losses.keys())
    losses = list(epoch_losses.values())

    plt.plot(epochs, losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Encoder Loss vs Epoch')
    plt.grid(True)
    plt.savefig("encoder_loss.png")

if __name__ == "__main__":
    input_file = "encoder.log"  # Provide the path to your file
    epoch_losses = process_file(input_file)
    plot_loss_vs_epoch(epoch_losses)
