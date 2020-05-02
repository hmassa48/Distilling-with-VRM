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

Train a set of techer models with these VRM techniques.

## Step 3: Train Student Models

Use dark knowledge from teacher models trained in Step 2.

## Step 4: Analyse generalization performance

Use different datasets and performance metrics to analyse generalization performance of the different student models. To measure generalization, we can evaluate the models on the unseen CIFAR test set. In addition to that, we also consider the following datasets:

- CIFAR 10.1 v6: Small natural variations in the dataset
- CINIC (ImageNet Fold): Distributional shift in images
- CIFAR 10H: CIFAR Test Set but with human labels - can help us in analysing prediction structure. 
