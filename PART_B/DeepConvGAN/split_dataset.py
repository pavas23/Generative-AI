import os
import random
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import shutil
from torch.utils.data import Dataset

def read_split_annotations(file_path):
    split_annotations = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            image_name = parts[0]
            split_label = int(parts[1])
            split_annotations[image_name] = split_label
    return split_annotations

def split_dataset(root_dir, partition_file):
    split_annotations = read_split_annotations(os.path.join(root_dir, partition_file))

    train_img_names = [image_name for image_name, split_label in split_annotations.items() if split_label == 0]
    val_img_names = [image_name for image_name, split_label in split_annotations.items() if split_label == 1]
    test_img_names = [image_name for image_name, split_label in split_annotations.items() if split_label == 2]

    # shuffling indices
    random.shuffle(train_img_names)
    random.shuffle(val_img_names)
    random.shuffle(test_img_names)

    return train_img_names, val_img_names, test_img_names

def copy_images(source_dir, image_names, destination_dir):
    for image_name in image_names:
        print(image_name)
        source_path = os.path.join(source_dir, image_name)
        print(source_path)
        destination_path = os.path.join(destination_dir,image_name)
        print(destination_path)
        shutil.copyfile(source_path, destination_path)

root_dir = ""
partition_file = "./list_eval_partition.txt"
train_img_names, val_img_names, test_img_names = split_dataset(root_dir,partition_file)

source_dir = "/mnt/MIG_store/Datasets/celeba/img_align_celeba/img_align_celeba"
os.makedirs("dataset", exist_ok=True)
destination_dir_train = "./dataset/celeba_train/img_align_celeba"
destination_dir_val = "./dataset/celeba_val/img_align_celeba"
destination_dir_test = "./dataset/celeba_test/img_align_celeba"

os.makedirs(destination_dir_train, exist_ok=True)
os.makedirs(destination_dir_val, exist_ok=True)
os.makedirs(destination_dir_test, exist_ok=True)

copy_images(source_dir,train_img_names,destination_dir_train)
copy_images(source_dir,val_img_names,destination_dir_val)
copy_images(source_dir,test_img_names,destination_dir_test)


class CelebADataset(Dataset):
    def __init__(self, mode, data_dir="/mnt/MIG_store/Datasets/celeba"):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        )

        self.df = pd.read_csv(os.path.join(data_dir, "list_attr_celeba.csv"))
        self.split = pd.read_csv(os.path.join(data_dir, "list_eval_partition.csv"))

        if mode == "men_with_glasses":
            self.df = self.df[(self.df["Male"] == 1) & (self.df["Eyeglasses"] == 1)]
        elif mode == "men_no_glasses":
            self.df = self.df[(self.df["Male"] == 1) & (self.df["Eyeglasses"] == -1)]
        elif mode == "women_no_glasses":
            self.df = self.df[(self.df["Male"] == -1) & (self.df["Eyeglasses"] == -1)]
        elif mode == "people_with_glasses":
            self.df = self.df[(self.df["Eyeglasses"] == 1)]
        elif mode == "people_no_glasses":
            self.df = self.df[(self.df["Eyeglasses"] == -1)]
        elif mode == "men_with_smile":
            self.df = self.df[(self.df["Male"] == 1) & (self.df["Smiling"] == 1)]
        elif mode == "people_with_hat":
            self.df = self.df[(self.df["Wearing_Hat"] == 1)]
        elif mode == "people_no_hat":
            self.df = self.df[(self.df["Wearing_Hat"] == -1)]
        elif mode == "people_with_mus":
            self.df = self.df[(self.df["Mustache"] == 1)]
        elif mode == "people_no_mus":
            self.df = self.df[(self.df["Mustache"] == -1)]

        self.image_paths = [
            os.path.join(data_dir, "img_align_celeba", "img_align_celeba", f"{img_id}")
            for img_id in self.df["image_id"]
            if self.split[self.split["image_id"] == img_id]["partition"].values[0] == 0
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
men_no_glasses = [CelebADataset("men_no_glasses").__getitem__(i) for i in range(5)]
people_with_glasses = [
    CelebADataset("people_with_glasses").__getitem__(i) for i in range(5)
]
people_no_glasses = [
    CelebADataset("people_no_glasses").__getitem__(i) for i in range(5)
]
men_with_glasses = [CelebADataset("men_with_glasses").__getitem__(i) for i in range(5)]
women_no_glasses = [CelebADataset("women_no_glasses").__getitem__(i) for i in range(5)]
men_with_smile = [CelebADataset("men_with_smile").__getitem__(i) for i in range(5)]
people_with_hat = [CelebADataset("people_with_hat").__getitem__(i) for i in range(5)]
people_no_hat = [CelebADataset("people_no_hat").__getitem__(i) for i in range(5)]
people_with_mus = [CelebADataset("people_with_mus").__getitem__(i) for i in range(5)]
people_no_mus = [CelebADataset("people_no_mus").__getitem__(i) for i in range(5)]