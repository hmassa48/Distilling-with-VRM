# Knowledge Distillation with VRM

This project aims to analyse the impact of various VRM techniques(applied on teacher models) on the generalization performance of a student model. The VRM techniques being analysed here are:

![alt text](https://raw.githubusercontent.com/deepandas11/Distilling-with-VRM/master/img/readme_image.png)

## Step 1: Replicate Conda Environment

```bash
conda create -n ml
conda install --name ml --file spec-file.txt
conda activate ml
```

## Step 2: Train Teacher Models

### For Baseline Teacher Training (Default Model: ResNet18)

```
python training_template.py --mode teacher --augmentation False --name resnet18_baseline
```

### For Teacher Training with Augmentation

```
python training_template.py --mode teacher --augmentation True --name resnet18_augmented
```

### For Mixup Training

```
python training_template.py --mode teacher --augmentation False --mixup True --name resnet18_mixup
```

### For Cutout Training
```
python training_template.py --mode teacher --cutout True --name cutout
```


### For CutMix Training
```
python training_template.py --mode teacher --cutmix True --name cutmix --cutmix_prob 0.5 --cutmix_beta 1.0
```

