from utils import utils
import torch
from torch.autograd import Variable


def train(model, optimizer, loss_fn, acc_fn, dataloader, use_gpu, epoch, writer, mixup=False, alpha=1.0):

    model.train()

    losses = utils.AverageMeter()

    epoch_steps = len(dataloader)

    for i, (train_batch, label_batch) in enumerate(dataloader):
        niter = (epoch - 1)*epoch_steps + i
        if use_gpu:
            train_batch, label_batch = train_batch.cuda(), label_batch.cuda()

        if mixup:
            train_batch, label_batch_a, label_batch_b, lam = utils.mixup_data(
                train_batch, label_batch, alpha, use_gpu)
            train_batch, label_batch_a, label_batch_b = map(
                Variable, (train_batch, label_batch_a, label_batch_b))

            output_batch = model(train_batch)
            loss = utils.mixup_loss_fn(
                loss_fn, output_batch, label_batch_a, label_batch_b, lam)
            losses.update(loss.item())

        else:
            # To set grad to zero on these
            accuracies = utils.AverageMeter()
            train_batch, label_batch = map(
                Variable, (train_batch, label_batch))
            output_batch = model(train_batch)

            loss = loss_fn(output_batch, label_batch)
            accuracy = acc_fn(output_batch, label_batch)

            losses.update(loss.item())
            accuracies.update(accuracy.item())
            writer.add_scalar('data/stepwise_training_accuracy',
                              accuracies.val, niter)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(("Step: {}, Current Loss: {}, RunningLoss: {}").format(
            i, loss, losses.avg))
        writer.add_scalar('data/stepwise_training_loss', losses.val, niter)


    writer.add_scalar('data/training_loss', losses.avg, epoch)
    if not mixup:
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
