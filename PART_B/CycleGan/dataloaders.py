import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from PIL import Image

transform = transforms.Compose(
    [
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)

batch_size=32

class CelebADataset(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.df = pd.read_csv(os.path.join(data_dir, "list_attr_celeba.csv"))

        if mode == "men_no_glasses":
            self.df = self.df[(self.df["Male"] == 1) & (self.df["Eyeglasses"] == -1)]
        elif mode == "men_with_glasses":
            self.df = self.df[(self.df["Male"] == 1) & (self.df["Eyeglasses"] == 1)]
        elif mode == "women_with_glasses":
            self.df = self.df[(self.df["Male"] == -1) & (self.df["Eyeglasses"] == 1)]
        else:
            raise ValueError("Invalid mode")

        self.image_paths = [
            os.path.join(data_dir, "img_align_celeba", "img_align_celeba", f"{img_id}")
            for img_id in self.df["image_id"]
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

# Datasets

men_no_glasses_dataset = CelebADataset(
    "/mnt/MIG_store/Datasets/celeba", "men_no_glasses", transform=transform
)
men_with_glasses_dataset = CelebADataset(
    "/mnt/MIG_store/Datasets/celeba", "men_with_glasses", transform=transform
)
women_with_glasses_dataset = CelebADataset(
    "/mnt/MIG_store/Datasets/celeba", "women_with_glasses", transform=transform
)

# Dataloaders

print(f"{men_no_glasses_dataset.__getitem__(0).shape}")

men_no_glasses_loader = DataLoader(
    men_no_glasses_dataset, batch_size=batch_size, shuffle=True
)

print(f"Length of men_no_glasses_loader: {men_no_glasses_loader.__len__()}")

men_with_glasses_loader = DataLoader(
    men_with_glasses_dataset, batch_size=batch_size, shuffle=True
)

print(f"Length of men with glasses loader: {men_with_glasses_loader.__len__()}")

women_with_glasses_loader = DataLoader(
    women_with_glasses_dataset, batch_size=batch_size, shuffle=True
)

print(f"Length of women with glasses loader: {women_with_glasses_loader.__len__()}")

print(f"Finished initialising datasets and dataloaders")

# sample_batch = next(iter(men_no_glasses_loader))
# torchvision.utils.save_image(
#     sample_batch, "men_no_glasses_sample.png", nrow=8, normalize=True, padding=2
# )

# sample_batch = next(iter(men_with_glasses_loader))
# torchvision.utils.save_image(
#     sample_batch, "men_with_glasses_sample.png", nrow=8, normalize=True, padding=2
# )

# sample_batch = next(iter(women_with_glasses_loader))
# torchvision.utils.save_image(
#     sample_batch, "women_with_glasses_sample.png", nrow=8, normalize=True, padding=2
# )
