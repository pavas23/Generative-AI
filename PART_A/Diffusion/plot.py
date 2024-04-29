import matplotlib.pyplot as plt
with open("loss.log", "r") as f:
    data = f.readlines()

epoch_losses = {}

for line in data:
    parts = line.strip().split(",")
    epoch_str = (
        parts[0].split("[")[1].split("/")[0]
    )
    epoch = int(epoch_str)
    loss = float(parts[2].split(":")[1])

    if epoch not in epoch_losses:
        epoch_losses[epoch] = []
    epoch_losses[epoch].append(loss)

for epoch, losses in epoch_losses.items():
    average_loss = sum(losses) / len(losses)
    
    
# plot average loss for each epoch from the dictionary 

plt.plot(list(epoch_losses.keys()), list(epoch_losses.values()))
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.savefig("loss_plot.png")