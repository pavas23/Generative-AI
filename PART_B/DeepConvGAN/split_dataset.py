import os
import random
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import shutil

# make hash map with image name and label (for train, validation and split set)
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

# split dataset according to labels
def split_dataset(root_dir, partition_file):
    split_annotations = read_split_annotations(os.path.join(root_dir, partition_file))

    # splitting images into 3 separate sets
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
partition_file = "list_eval_partition.txt"
train_img_names, val_img_names, test_img_names = split_dataset(root_dir,partition_file)

source_dir = "/mnt/MIG_store/Datasets/celeba/img_align_celeba/img_align_celeba"
destination_dir_train = "./celeba_train"
destination_dir_val = "./celeba_val"
destination_dir_test = "./celeba_test"

# this will copy images to respective directories
copy_images(source_dir,train_img_names,destination_dir_train)
copy_images(source_dir,val_img_names,destination_dir_val)
copy_images(source_dir,test_img_names,destination_dir_test)
