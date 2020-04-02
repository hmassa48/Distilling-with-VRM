import os
import time
import torch
import argparse
from utils import utils, dataloader, model_fetch
import train
from tensorboardX import SummaryWriter


_CLASSES = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--seed', default=1)
parser.add_argument('--gpu', default=False)
parser.add_argument('--mode', default='teacher')
parser.add_argument('--teacher_model', default='resnet50')
parser.add_argument('--student_model', default='resnet50')
parser.add_argument('--augmentation', default=False)
parser.add_argument('--resume', default='')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--n_epochs', default=250)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--split_data', default=False)
parser.add_argument('--name', default='teacher_training_no_augmentation')
parser.add_argument('--mixup', default=False)
parser.add_argument('--alpha', default=1.0)


def main_teacher(args):

    print(
        ("Process {}, running on {}: starting {}").format(os.getpid(), os.name, time.asctime))
    print("Training with Mixup: ", args.mixup)
    process_num = round(time.time())
    dir_name = args.name + '_' + str(process_num)
    tb_path = "distillation_experiments/logs/%s/" % (dir_name)

    writer = SummaryWriter(tb_path)

    use_gpu = args.gpu
    if not torch.cuda.is_available():
        use_gpu = False

    # Load Models
    model = model_fetch.fetch_teacher(args.teacher_model)

    if use_gpu:
        model = model.cuda()

    train_loader = dataloader.fetch_dataloader(
        "train", args.augmentation, args.batch_size)
    test_loader = dataloader.fetch_dataloader(
        "test", args.augmentation, args.batch_size)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.99)
    loss_fn = utils.loss_fn
    acc_fn = utils.accuracy

    start_epoch, best_loss = utils.load_checkpoint(model, args.resume)
    epoch = start_epoch
    while epoch <= int(args.n_epochs):
        print("="*50)
        utils.adjust_learning_rate(args.lr, optimizer, epoch)
        print(("Epoch {} Training Starting").format(epoch))
        print("Learning Rate : ", utils.get_lr(optimizer))

        train_loss = train.train(
            model, optimizer, loss_fn, acc_fn, train_loader, use_gpu, epoch, writer, args.mixup, args.alpha)
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
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    if args.seed:
        torch.manual_seed(args.seed)

    if args.mode == 'teacher':
        main_teacher(args)
    elif args.mode == 'student':
        main_kd(args)
