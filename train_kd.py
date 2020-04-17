from utils import utils
import torch
from torch.autograd import Variable


def train(model, teacher_model, optimizer, loss_fn, dataloader, use_gpu, epoch, writer, temp=1.0, gamma=1.0):

    model.train()
    teacher_model.eval()

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    epoch_steps = len(dataloader)
    teacher_model.eval()

    for i, (train_batch, label_batch) in enumerate(dataloader):
        niter = (epoch - 1)*epoch_steps + i
        if use_gpu:
            train_batch, label_batch = train_batch.cuda(), label_batch.cuda()


        train_batch, label_batch = map(
            Variable, (train_batch, label_batch))

        output_batch = model(train_batch)
        teacher_output_batch = teacher_model(train_batch)
        teacher_output_batch = Variable(teacher_output_batch)

        loss = loss_fn(output_batch, label_batch, teacher_output_batch, temp, gamma)
        acc, _, _ = utils.find_metrics(output_batch, label_batch, use_gpu)

        losses.update(loss.item())
        accuracies.update(acc)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(("Step: {}, Current Loss: {}, RunningLoss: {}").format(
            i, loss, losses.avg))
        writer.add_scalar('data/stepwise_training_loss', losses.val, niter)
        writer.add_scalar(
                'data/stepwise_training_accuracy', accuracies.val, niter)

    writer.add_scalar('data/training_loss', losses.avg, epoch)
    writer.add_scalar('data/training_accuracy', accuracies.avg, epoch)


    return losses.avg


def validate(model, loss_fn, dataloader, use_gpu, epoch, writer):

    
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    precisions = utils.AverageMeter()
    recalls = utils.AverageMeter()

    model.eval()

    for i, (train_batch, label_batch) in enumerate(dataloader):
        if use_gpu:
            train_batch, label_batch = train_batch.cuda(), label_batch.cuda()

        with torch.no_grad():
            train_batch, label_batch = Variable(
                train_batch), Variable(label_batch)
            output_batch = model(train_batch)

            loss = loss_fn(output_batch, label_batch)
            acc, prec, rec = utils.find_metrics(output_batch, label_batch, use_gpu)

            losses.update(loss.item())
            accuracies.update(acc)
            precisions.update(prec)
            recalls.update(rec)

            print(("Step: {}, Current Loss: {}, RunningLoss: {}").format(
                i, loss, losses.avg))


    writer.add_scalar('data/val_loss', losses.avg, epoch)
    writer.add_scalar('data/val_accuracy', accuracies.avg, epoch)
    writer.add_scalar('data/val_precision', precisions.avg, epoch)
    writer.add_scalar('data/val_recall', recalls.avg, epoch)

    return losses.avg
