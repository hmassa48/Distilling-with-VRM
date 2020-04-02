import os
import shutil
import torch
import torch.nn as nn
import numpy as np



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val= val
        self.sum = sum(self.vals)
        self.count = len(self.vals)
        self.avg = self.sum/self.count


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Returns: (float) accuracy in [0,1]
    """
    outputs = outputs.detach().numpy()
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(outputs.shape[0])

    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(base_lr, optimizer, epoch, lr_decay=0.2):
    """Adjusts learning rate based on epoch of training
    
    At 60th, 120th and 150th epoch, divide learning rate by 0.2
    
    Args:
        base_lr: starting learning rate
        optimizer: optimizer used in training, SGD usually
        epoch: current epoch
        lr_decay: decay ratio (default: {0.2})
    """
    if epoch < 60:
        lr = base_lr
    elif 60 <= epoch < 120:
        lr = base_lr / 0.2
    elif 120 <= epoch < 150:
        lr = base_lr / (0.2)**2
    else:
        lr = base_lr / (0.2)**3

    # lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, dir_name, filename='checkpoint.pth.tar'):
    """Save model at end of each epoch and best model if found
    
    Saves state of the model at end of each epoch and the best model
    
    Args:
        state: dictionary with epoch, best loss and model information
        is_best: boolean if current model is the best till now
        dir_name: directory for model paths for this model
        filename: epoch filename (default: {'checkpoint.pth.tar'})
    """
    directory = "distillation_experiments/runs/%s/" % (dir_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = directory + filename
    torch.save(state, filename)
    print("Saved Checkpoint!")

    if is_best:
        print("Best Model found ! ")
        shutil.copyfile(filename, directory + '/model_best.pth.tar')


def load_checkpoint(model, resume_filename):
    """Load a model at the state for resume_filename

    Loads trained model parameters onto model
    
    Args:
        model: empty model
        resume_filename: using resume path
    
    Returns:
        start_epoch: Epoch at which model at resume_filename was last trained
        best_loss: Best Loss at start_epoch
    """
    start_epoch = 1
    best_loss = float("inf")

    if resume_filename:
        if os.path.isfile(resume_filename):
            print("=> Loading Checkpoint '{}'".format(resume_filename))
            checkpoint = torch.load(resume_filename)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['model'])

            print("========================================================")

            print("Loaded checkpoint '{}' (epoch {})".format(
                resume_filename, checkpoint['epoch']))
            print("Current Loss : ", checkpoint['best_loss'])

            print("========================================================")

        else:
            print(" => No checkpoint found at '{}'".format(resume_filename))

    return start_epoch, best_loss

def mixup_data(x, y, alpha=1.0, use_gpu=True):
    """Create mixup data

    Generates convex combination of datapoints

    Args:
        x: batch of images
        y: batch of labels
        alpha: parameter controlling lambda (default: {1.0})
        use_gpu: if using gpu  (default: {True})

    Returns:
        mixed_up: convex sum of data-points
        y_a: original labels
        y_b: other labels
        lam: combination parameter
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if use_gpu:
        index = index.cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_loss_fn(loss_fn, pred, y_a, y_b, lam):
    """Loss function for mixup
  
    Args:
        loss_fn: Original Loss function
        pred: Predictions
        y_a: Data point A label
        y_b: Data point B label
        lam: combination parameter
    
    Returns:
        loss
    """
    return lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)

