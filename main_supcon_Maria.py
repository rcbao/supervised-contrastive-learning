from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from collections import Counter
from torch.utils.data import Dataset


from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
import numpy as np
# from main_ce import CustomBrainDataset
from PIL import Image


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

data_dir = "Classification_AD_CN_MCI_datasets"

final_X_train = np.load(f'{data_dir}/brain_train_image_final.npy')
final_y_train = np.load(f'{data_dir}/brain_train_label.npy')

final_X_train_modified = final_X_train[:, 1, :, :]

# Print data statistics early
print("\nInitial Dataset Statistics:")
print(f"Number of training images: {len(final_X_train_modified)}")
# print(f"Number of test images: {len(final_X_test_modified)}")

train_labels_count = Counter(final_y_train)
# test_labels_count = Counter(final_y_test)

print("Training Dataset Label Counts:")
for label, count in sorted(train_labels_count.items()):
    print(f"  Label {label}: {count} images")

# print("Test Dataset Label Counts:")
# for label, count in sorted(test_labels_count.items()):
#     print(f"  Label {label}: {count} images")


class CustomBrainDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # shape: (H, W)
        # Convert to PIL Image in grayscale mode
        img = Image.fromarray(img.astype(np.uint8), mode='L')

        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        print()
        # assert opt.data_folder is not None \
        #     and opt.mean is not None \
        #     and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_cls = 3
    
    return opt


def set_loader(opt):
    # construct data loader
    # if opt.dataset == 'cifar10':
    #     mean = (0.4914, 0.4822, 0.4465)
    #     std = (0.2023, 0.1994, 0.2010)
    # elif opt.dataset == 'cifar100':
    #     mean = (0.5071, 0.4867, 0.4408)
    #     std = (0.2675, 0.2565, 0.2761)
    # elif opt.dataset == 'path':
    #     mean = eval(opt.mean)
    #     std = eval(opt.std)
    # else:
    #     raise ValueError('dataset not supported: {}'.format(opt.dataset))

    mean = (0.00143,)
    std = (0.00149,)
    normalize = transforms.Normalize(mean=mean, std=std)

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    #     ], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,])



    # if opt.dataset == 'cifar10':
    #     train_dataset = datasets.CIFAR10(root=opt.data_folder,
    #                                      transform=TwoCropTransform(train_transform),
    #                                      download=True)
    # elif opt.dataset == 'cifar100':
    #     train_dataset = datasets.CIFAR100(root=opt.data_folder,
    #                                       transform=TwoCropTransform(train_transform),
    #                                       download=True)
    # elif opt.dataset == 'path':
    #     train_dataset = datasets.ImageFolder(root=opt.data_folder,
    #                                         transform=TwoCropTransform(train_transform))
    # else:
    #     raise ValueError(opt.dataset)

    transform = transforms.Compose([
        # Adjust or simplify augmentations if needed
        # transforms.RandomResizedCrop((80, 60)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.00143], std=[0.00149]),
    ])

    train_dataset = CustomBrainDataset(final_X_train_modified, final_y_train, transform=TwoCropTransform(train_transform))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg

def get_augmented_statistics(data_loader):
    augmented_labels = []
    
    # Iterate over the data loader
    for _, labels in data_loader:
        augmented_labels.extend(labels.tolist())  # Collect all labels after augmentation

    # Count the occurrences of each label
    augmented_labels_count = Counter(augmented_labels)

    print("\nAugmented Dataset Statistics:")
    print(f"Number of augmented images: {len(augmented_labels)}")

    print("Augmented Dataset Label Counts:")
    for label, count in sorted(augmented_labels_count.items()):
        print(f"  Label {label}: {count} images")


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # Analyze dataset statistics after augmentation
    print("\nDataset Statistics After Augmentation:")
    get_augmented_statistics(train_loader)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # Early stopping parameters
    patience = 5  # Number of epochs to wait before stopping
    min_delta = 1e-4  # Minimum improvement in loss to be considered significant
    best_loss = float('inf')  # Initialize with a very high loss value
    patience_counter = 0  # Counter for tracking epochs without improvement

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Check for improvement in loss
        if best_loss - loss > min_delta:
            best_loss = loss  # Update the best loss
            patience_counter = 0  # Reset the patience counter
        else:
            patience_counter += 1  # Increment the patience counter

        print(f"Loss value: {loss:.6f} -- Epoch: {epoch} -- Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Save model checkpoint
        if epoch % opt.save_freq == 0 or patience_counter == 0:  # Save if improvement occurs
            save_file = os.path.join(
                opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)

        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Save the final model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
