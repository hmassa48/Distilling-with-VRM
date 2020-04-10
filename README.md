# Knowledge Distillation with VRM

First, replicate the conda environment using the spec-file and activate it.

```bash
conda create -n ml
conda install --name ml --file spec-file.txt
conda activate ml
```

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

