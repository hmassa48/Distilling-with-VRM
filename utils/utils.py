import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


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
        self.val = val
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
    labels = labels.detach().numpy()
    outputs = np.argmax(outputs, axis=1)

    return np.sum(outputs == labels)/float(outputs.size)

def check_type(outputs, labels, use_gpu):
    if type(outputs) is not np.ndarray:
        if use_gpu:
            outputs = outputs.cpu()
        outputs = outputs.detach().numpy()

    if type(labels) is not np.ndarray:
        if use_gpu:
            labels = labels.cpu()
        labels = labels.detach().numpy()

    outputs = np.argmax(outputs, axis=1)
    return outputs, labels


def find_metrics(outputs, labels, use_gpu):
    outputs, labels = check_type(outputs, labels, use_gpu)
    prec, rec, f_score, _ = precision_recall_fscore_support(
        labels, outputs, average='weighted')
    accuracy = np.sum(outputs == labels) / float(outputs.size)
    return accuracy, prec, rec, f_score


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(base_lr, optimizer, epoch, lr_decay=0.1):
    """Adjusts learning rate based on epoch of training

    At 60th, 80th and 120th epoch, divide learning rate by 0.2

    Args:
        base_lr: starting learning rate
        optimizer: optimizer used in training, SGD usually
        epoch: current epoch
        lr_decay: decay ratio (default: {0.2})
    """
    if epoch < 60:
        lr = base_lr
    elif 60 <= epoch < 80:
        lr = base_lr * lr_decay
    elif 80 <= epoch < 120:
        lr = base_lr * (lr_decay)**2
    else:
        lr = base_lr * (lr_decay)**3

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
            if not torch.cuda.is_available():
                checkpoint = torch.load(resume_filename, map_location=torch.device('cpu'))
            else:
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


def cutmix_data(x, y, beta=1.0, cutmix_prob=0.5, use_gpu=True):
    if beta > 0 and cutmix_prob < 1.0:
        lam = np.random.beta(beta, beta)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        if use_gpu:
            index = index.cuda()

        y_a = y
        y_b = y[index]

        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                   (x.size()[-1] * x.size()[-2]))

        return x, y_a, y_b, lam

    else:
        lam = 1
        return x, y, y, lam


def mixed_loss_fn(loss_fn, pred, y_a, y_b, lam):
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


def kd_loss_fn(output, label, teacher_output, temp=1.0, gamma=1.0):

    loss = nn.KLDivLoss()(F.log_softmax(output/temp, dim=1),
                          F.softmax(teacher_output/temp, dim=1)) * (gamma * temp * temp) + \
        F.cross_entropy(output, label) * (1 - gamma)

    return loss


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
