import matplotlib.pyplot as plt
import re

def parse_losses(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    epochs = []
    gen_losses = []
    disc_losses = []
    d_x_array = []
    d_g_z_array = []
    curr_epoch = 0
    epoch_d_avg = 0
    epoch_g_avg = 0
    epoch_d_x_avg = 0
    epoch_d_g_z_avg = 0
    count = 0

    for line in lines:
        match = re.match(r"\[(\d+)/(\d+)\]\[(\d+)/(\d+)\]\s+Loss_D:\s+([\d.]+)\s+Loss_G:\s+([\d.]+)\s+D\(x\):\s+([\d.]+)\s+D\(G\(z\)\):\s+([\d.]+) / ([\d.]+)", line)
        if match:
            epoch = int(match.group(1))
            loss_d = float(match.group(5))
            loss_g = float(match.group(6))
            d_x = float(match.group(7))
            d_g_z = float(match.group(8))
            if epoch == curr_epoch:
                epoch_d_avg += loss_d
                epoch_g_avg += loss_g
                epoch_d_x_avg += d_x
                epoch_d_g_z_avg += d_g_z
                count += 1
            else:
                epochs.append(epoch)
                gen_losses.append(epoch_g_avg/count)
                disc_losses.append(epoch_d_avg/count)
                d_x_array.append(epoch_d_x_avg/count)
                d_g_z_array.append(epoch_d_g_z_avg/count)
                count = 1
                curr_epoch += 1
                epoch_g_avg = loss_g
                epoch_d_avg = loss_d
                epoch_d_x_avg = d_x
                epoch_d_g_z_avg = d_g_z
    return epochs, gen_losses, disc_losses, d_x_array, d_g_z_array

def plot_losses(log_file):
    epochs, gen_losses, disc_losses, d_x_array, d_g_z_array = parse_losses(log_file)

    plt.plot(epochs,gen_losses,label='Generator Loss', color='blue')
    plt.plot(epochs,disc_losses,label='Discriminator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Losses for DCGAN')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss1.png")

def plot_probabilities(log_file):
    epochs, gen_losses, disc_losses, d_x_array, d_g_z_array = parse_losses(log_file)

    plt.close()
    plt.plot(epochs,d_x_array,label='D(x)',color='orange')
    plt.plot(epochs,d_g_z_array,label='D(G(z))',color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.title('Generator and Discriminator Probabilities for DCGAN')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss2.png")


if __name__ == "__main__":
    log_file = 'out.log'
    plot_losses(log_file)
    plot_probabilities(log_file)
