from tqdm import tqdm
from utils import utils
import torch
from torch.autograd import Variable


def train(model, optimizer, loss_fn, acc_fn, dataloader, use_gpu, epoch, writer):

    model.train()

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    epoch_steps = len(dataloader)

    for i, (train_batch, label_batch) in enumerate(dataloader):
        if use_gpu:
            train_batch, label_batch = train_batch.cuda(), label_batch.cuda()

        # To set grad to zero on these
        train_batch, label_batch = Variable(train_batch), Variable(label_batch)
        output_batch = model(train_batch)

        loss = loss_fn(output_batch, label_batch)
        accuracy = acc_fn(output_batch, label_batch)

        losses.update(loss.item())
        accuracies.update(accuracy.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(("Step: {}, Current Loss: {}, RunningLoss: {}").format(
            i, loss, losses.avg))

        niter = (epoch - 1)*epoch_steps + i

        writer.add_scalar('data/stepwise_training_loss', losses.val, niter)
        writer.add_scalar('data/stepwise_training_accuracy', accuracies.val, niter)


    writer.add_scalar('data/training_loss', losses.avg, epoch)
    writer.add_scalar('data/training_accuracy', accuracies.avg, epoch)

    return losses.avg


def validate(model, loss_fn, acc_fn, dataloader, use_gpu, epoch, writer):

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    model.eval()

    for i, (train_batch, label_batch) in enumerate(dataloader):
        if use_gpu:
            train_batch, label_batch = train_batch.cuda(), label_batch.cuda()

        with torch.no_grad():
            train_batch, label_batch = Variable(
                train_batch), Variable(label_batch)
            output_batch = model(train_batch)

            loss = loss_fn(output_batch, label_batch)
            accuracy = acc_fn(output_batch, label_batch)

            losses.update(loss.item())
            accuracies.update(accuracy.item())

            print(("Step: {}, Current Loss: {}, RunningLoss: {}").format(
                i, loss, losses.avg))

    writer.add_scalar('data/val_loss', losses.avg, epoch)
    writer.add_scalar('data/val_accuracy', accuracies.avg, epoch)

    return losses.avg
