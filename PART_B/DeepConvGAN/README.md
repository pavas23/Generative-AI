## Steps to Run

```pip install -r requirements.txt```

```python
mkdir /dataset
dir structure should be:
  /dataset
    /celeba
      /img_align_celeba
    /celeba_train
      /img_align_celeba
    /celeba_val
      /img_align_celeba
    /celeba_test
      /img_align_celeba
    list_attr_celeba.txt
    list_eval_partition.txt
```
download the celeba dataset inside the ```/dataset/celeba``` dir

make ```list_attr_celeba.txt``` and ```list_eval_partition.txt```

run ```python split_dataset.py``` to populate train, val and test dirs

run ```python train.py``` to train the dcgan model
