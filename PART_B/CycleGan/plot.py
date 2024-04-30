import matplotlib.pyplot as plt
import re

def parse_losses(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    epochs = []
    gen_losses = []
    disc_losses = []
    

    for line in lines:
        # match = re.match(r"\[(\d+)/(\d+)\]\[(\d+)/(\d+)\]\s+Loss_D:\s+([\d.]+)\s+Loss_G:\s+([\d.]+).*", line)
        match = re.match(r"Epoch \[(\d+)/(\d+)\], Loss_G: ([\d.]+), Loss_D: ([\d.]+)", line)

        if match:
            epoch = int(match.group(1))
            loss_g = float(match.group(3))
            loss_d = float(match.group(4))
            
            epochs.append(epoch)
            gen_losses.append(loss_g)
            disc_losses.append(loss_d)
            
                
    return epochs, gen_losses, disc_losses

def plot_losses(name,log_file):
    epochs, gen_losses, disc_losses = parse_losses(log_file)
    
    plt.plot(epochs,gen_losses,label='Generator Loss', color='blue')
    plt.plot(epochs,disc_losses,label='Discriminator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Losses for {name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_file}.png")
    plt.close()




if __name__ == "__main__":
    plot_losses("Men without glasses and men with glasses","loss_men.log")
    plot_losses("Men with glasses and women with glasses","loss_women.log")
    
