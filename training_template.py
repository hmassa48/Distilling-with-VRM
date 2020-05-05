import os
import time
import torch
import argparse
from utils import utils, dataloader, model_fetch, cutout
import train
import train_kd
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from pprint import pprint

_CLASSES = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# General Arguments
parser.add_argument('--seed', default=2)
parser.add_argument('--gpu', default=True)
parser.add_argument('--mode', default='teacher')
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--lr_decay', default=0.1)
parser.add_argument('--n_epochs', default=150)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--decay', default=1e-4) # Weight Decay
parser.add_argument('--dataset', default='mnist')

# Model arguments
parser.add_argument('--teacher_model', default='alexnet')
parser.add_argument('--student_model', default='lenet')
parser.add_argument('--resume', default='')
parser.add_argument('--teacher_path', default='')

# VRM arguments
parser.add_argument('--cutout', default=False)
parser.add_argument('--augmentation', default=True)
parser.add_argument('--mixup', default=False)
parser.add_argument('--cutmix', default=False)

# VRM parameters
parser.add_argument('--alpha', default=1.0)   # Mixup Alpha
parser.add_argument('--n_holes', default=1)   # Cutout
parser.add_argument('--length_holes', default=16)  # Cutout
parser.add_argument('--cutmix_beta', default=1.0)  
parser.add_argument('--cutmix_prob', default=0.5)

# Distillation arguments
parser.add_argument('--temperature', default=5.0)
parser.add_argument('--gamma', default=0.5)

# Name argument
parser.add_argument('--name', default='alexnet_augmented')

def main_teacher(args):

    print(
        ("Process {}, running on {}: starting {}").format(os.getpid(), os.name, time.asctime))

    print("Training with Augmentation: ", args.augmentation)
    print("Training with Cutout: ", args.cutout)
    print("Training with Mixup: ", args.mixup)
    print("Training with CutMix: ", args.cutmix)

    process_num = round(time.time())
    dir_name = args.name + '_' + str(process_num) + str(args.dataset)
    tb_path = "distillation_experiments/logs/%s/" % (dir_name)
    pprint(args.__dict__)

    writer = SummaryWriter(tb_path)

    use_gpu = args.gpu
    if not torch.cuda.is_available():
        use_gpu = False

    # Load Models
    model = model_fetch.fetch_teacher(args.teacher_model)

    if use_gpu:
        cudnn.benchmark = True
        model = model.cuda()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train_transform = val_transform = transform

    if args.augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    if args.cutout:
        train_transform.transforms.append(cutout.Cutout(
            n_holes=args.n_holes, length=args.length_holes))

    train_loader = dataloader.fetch_dataloader(
        "train", train_transform, args.dataset, args.batch_size)
    test_loader = dataloader.fetch_dataloader(
        "test", val_transform, args.dataset, args.batch_size)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=args.decay)
    loss_fn = utils.loss_fn
    acc_fn = utils.accuracy

    start_epoch, best_loss = utils.load_checkpoint(model, args.resume)
    epoch = start_epoch
    while epoch <= int(args.n_epochs):
        print("="*50)
        utils.adjust_learning_rate(args.lr, optimizer, epoch, args.lr_decay)
        print(("Epoch {} Training Starting").format(epoch))
        print("Learning Rate : ", utils.get_lr(optimizer))

        train_loss = train.train(
            model, optimizer, loss_fn, acc_fn, train_loader, use_gpu,
            epoch, writer, args.mixup, args.alpha, args.cutmix,
            args.cutmix_prob, args.cutmix_beta
        )

        val_loss = train.validate(
            model, loss_fn, acc_fn, test_loader, use_gpu, epoch, writer)

        print("-"*50)
        print(
            ("Epoch {}, Training-Loss: {}, Validation-Loss: {}").format(epoch, train_loss, val_loss))
        print("="*50)

        curr_state = {
            "epoch": epoch,
            "best_loss": min(best_loss, val_loss),
            "model": model.state_dict()
        }

        # # Use only if model to be saved at each epoch
        # filename = 'epoch_' + str(epoch) + '_checkpoint.pth.tar'

        utils.save_checkpoint(
            state=curr_state,
            is_best=bool(val_loss < best_loss),
            dir_name=dir_name,
            # filename=filename
        )

        if val_loss < best_loss:
            best_loss = val_loss
        epoch += 1
        writer.add_scalar('data/learning_rate', utils.get_lr(optimizer), epoch)


def main_kd(args):

    print(
        ("Process {}, running on {}: starting {}").format(os.getpid(), os.name, time.asctime))

    print("Student Training Underway!")
    print("Temperature: ", args.temperature)
    print("Relative Loss Weights: ", args.gamma)
    process_num = round(time.time())

    model_name = args.name + '_temp' + \
        str(args.temperature) + '_gamma' + str(args.gamma)
    dir_name = model_name + '_' + str(process_num)

    tb_path = "distillation_experiments/logs/%s/" % (dir_name)

    writer = SummaryWriter(tb_path)

    print("Arguments for model: ", model_name)
    pprint(args.__dict__)

    use_gpu = args.gpu
    if not torch.cuda.is_available():
        use_gpu = False

    # Load Models
    teacher_model = model_fetch.fetch_teacher(args.teacher_model)
    student_model = model_fetch.fetch_student(args.student_model)

    if use_gpu:
        cudnn.benchmark = True
        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train_transform = val_transform = transform

    if False:
        if args.augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        if args.cutout:
            train_transform.transforms.append(cutout.Cutout(
                n_holes=args.n_holes, length=args.length_holes))

    train_loader = dataloader.fetch_dataloader(
        "train", train_transform, args.batch_size)
    test_loader = dataloader.fetch_dataloader(
        "test", val_transform, args.batch_size)

    params = [p for p in student_model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.99)

    loss_fn = utils.kd_loss_fn
    simple_loss_fn = utils.loss_fn

    teacher_epoch, teacher_loss = utils.load_checkpoint(
        teacher_model, args.teacher_path)
    start_epoch, best_loss = utils.load_checkpoint(student_model, args.resume)

    print("Models Loaded!")
    print(student_model)
    
    epoch = start_epoch
    while epoch <= int(args.n_epochs):
        print("="*50)
        utils.adjust_learning_rate(args.lr, optimizer, epoch, args.lr_decay)
        print(("Epoch {} Training Starting").format(epoch))
        print("Learning Rate : ", utils.get_lr(optimizer))

        train_loss = train_kd.train(
            student_model, teacher_model, optimizer, loss_fn, train_loader,
            use_gpu, epoch, writer, args.temperature, args.gamma
        )
        val_loss = train_kd.validate(
            student_model, simple_loss_fn, test_loader, use_gpu, epoch, writer)

        print("-"*50)
        print(
            ("Epoch {}, Training-Loss: {}, Validation-Loss: {}").format(epoch, train_loss, val_loss))
        print("="*50)

        curr_state = {
            "epoch": epoch,
            "best_loss": min(best_loss, val_loss),
            "model": student_model.state_dict()
        }

        # # Use only if model to be saved at each epoch
        # filename = 'epoch_' + str(epoch) + '_checkpoint.pth.tar'

        utils.save_checkpoint(
            state=curr_state,
            is_best=bool(val_loss < best_loss),
            dir_name=dir_name,
            # filename=filename
        )

        if val_loss < best_loss:
            best_loss = val_loss
        epoch += 1
        writer.add_scalar('data/learning_rate', utils.get_lr(optimizer), epoch)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.seed:
        torch.manual_seed(args.seed)

    if args.mode == 'teacher':
        main_teacher(args)
    elif args.mode == 'student':
        main_kd(args)
