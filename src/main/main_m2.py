import os
import time
import shutil
import argparse
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import my_models as mm

writer = None
best_acc1 = 0

parser = argparse.ArgumentParser(description='MyResNet18 (M2) Training and Testing')
parser.add_argument('data',
                    metavar='DIR',
                    nargs='?',
                    default='tiny_imagenet_200',
                    help='path to dataset (default: tiny_imagenet_200)')
parser.add_argument('--writer',
                    default='tiny_imagenet_200_m2',
                    help='filefolder name of tensorboard (default: tiny_imagenet_200_m2)')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default 0)')
parser.add_argument('--epochs', default=75, type=int, help='number of total epochs to run')
parser.add_argument('-nc', '--num-classes', type=int, default=200, help='class num')
parser.add_argument('--gamma', type=float, default=0.1, help='StepLR gamma')
parser.add_argument('--step-size', type=int, default=30, help='StepLR step-size')
parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('-b', '--batch-size', type=int, default=128, help='mini-batch size (default 128)')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-p', '--print-freq', default=70, type=int, metavar='N', help='print frequency (default: 70)')
parser.add_argument('--write-graph', action='store_true', help='write graph of structure on tensorboard')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")


def main():
    print('ResNet18 M2, without normalization')
    args = parser.parse_args()
    global writer
    global best_acc1
    writer = SummaryWriter('runs/' + args.writer)

    # create model
    model = mm.MyResNet18M2(num_classes=args.num_classes)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print('using CPU, this will be slow')
    else:
        # device = torch.device('cuda')  # this will return current cuda device, see pytorch docs for details
        device = torch.device('cuda', args.gpu)
        model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Load data
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(100000, (3, 224, 224), args.num_classes, transforms.ToTensor())
        val_dataset = datasets.FakeData(10000, (3, 224, 224), args.num_classes, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        trans = transforms.Compose([
            transforms.Resize((64, 64)),  # not must 224
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(traindir, trans)
        val_dataset = datasets.ImageFolder(valdir, trans)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, -1, device)
        return

    if args.write_graph:
        print('Writing graph')
        my_dataiter = iter(train_loader)
        imgs, _ = my_dataiter.next()
        writer.add_graph(model, imgs)
        writer.flush()
        return

    for epoch in range(args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, epoch, device)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best)

    writer.flush()


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    global writer
    running_loss = 0

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        running_loss += loss.item()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

        if i % args.print_freq == args.print_freq - 1:
            writer.add_scalar('training loss (M2)', running_loss / args.print_freq, epoch * len(train_loader) + i + 1)
            writer.add_scalar('training acc1 (M2)', top1.avg, epoch * len(train_loader) + i + 1)
            writer.add_scalar('training acc5 (M2)', top5.avg, epoch * len(train_loader) + i + 1)
            running_loss = 0


def validate(val_loader, model, criterion, args, epoch, device):
    def run_validate(loader):
        validate_loss = 0
        global writer

        with torch.no_grad():
            end = time.time()

            for i, (images, target) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)
                validate_loss += loss.item()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

            if epoch != -1:
                writer.add_scalar('validate loss (M2)', validate_loss / len(loader), epoch)

    global writer
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    progress.display_summary()
    writer.add_scalar('validate acc1 (M2)', top1.avg, epoch)
    writer.add_scalar('validate acc5 (M2)', top5.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint_m2.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_m2.pth.tar')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
