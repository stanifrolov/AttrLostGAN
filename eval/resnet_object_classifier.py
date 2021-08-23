from __future__ import print_function
from __future__ import division

import argparse
import datetime
from ast import literal_eval
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models, transforms
import time
import subprocess
import os
import copy
import pandas as pd
import random
from PIL import Image
from utils.logger import setup_logger

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_recent_commit():
    return str(subprocess.check_output(["git", "describe", "--always"]).strip().decode())


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


def train_model(model, dataloaders, criterion, optimizer, logger, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            df = pd.DataFrame()

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # save predictions
                if phase in ['val', 'test']:
                    for i in range(len(labels)):
                        d = {
                            'target': np.asarray(labels[i].data.cpu()),
                            'prediction': np.asarray(preds[i].data.cpu()),
                        }

                        df = df.append(d, ignore_index=True)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            if phase in ['val', 'test']:
                df = df[['target', 'prediction']]
                df.to_csv(os.path.join(args.out_path, phase + '_predictions', 'epoch_%d.csv' % (epoch + 1)),
                          index=False)

        torch.save(model.state_dict(),
                   os.path.join(args.out_path, 'model/', 'model_%d.pth' % (epoch + 1)))

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def initialize_model(model_name, num_classes, feature_extract, use_pretrained):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnet101-cifar":
        model_ft = ResNet101(num_classes=num_classes)
        init_params(model_ft)
        input_size = 32
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def get_recent_commit():
    return str(subprocess.check_output(["git", "describe", "--always"]).strip().decode())


class LoadSamples(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=0)
        self.root_dir = root_dir
        self.transform = transform

        self.annotations['label'] = self.annotations['label'].apply(literal_eval)
        self.annotations['label_name'] = self.annotations['label_name'].apply(literal_eval)
        self.indices = list(
            self.annotations[self.annotations['label_name'].map(lambda d: d) != "__image__"]['label_name'].index)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        img_name = os.path.join(self.root_dir, self.annotations.loc[idx, 'name'])
        image = Image.open(img_name).convert('RGB')

        labels = self.annotations.loc[idx, 'label'][0] - 1  # labels is a single element list

        if self.transform is not None:
            image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.int64)
        assert labels >= 0, (labels, self.annotations.loc[idx, 'name'], self.annotations.loc[idx, 'label_name'])

        return image, labels


def main(args):
    # get most recent commit
    commit_obj = get_recent_commit()
    current_time = time.strftime("%d-%m-%Y_%H_%M_%dd_%mm", time.localtime())

    # make dirs
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.makedirs(os.path.join(args.out_path, 'model/'))
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'val_predictions/')):
        os.makedirs(os.path.join(args.out_path, 'val_predictions/'))
    if not os.path.exists(os.path.join(args.out_path, 'test_predictions/')):
        os.makedirs(os.path.join(args.out_path, 'test_predictions/'))

    logger = setup_logger("CAS Evaluation", args.out_path, 0,
                          filename=current_time + '_log.txt')
    logger.info('Commit Tag: ' + commit_obj)
    logger.info('Time: ' + current_time)
    logger.info(args.message)

    [logger.info(arg + " : " + str(getattr(args, arg))) for arg in vars(args)]
    logger.info(args)
    logger.info("PyTorch Version: " + torch.__version__)
    logger.info("Torchvision Version: " + torchvision.__version__)

    start_time = time.time()

    # Initialize the model for this run
    model_ft, input_size = initialize_model(args.model_name,
                                            args.num_classes,
                                            args.feature_extract,
                                            args.use_pretrained)

    # Print the model we just instantiated
    # print(model_ft)

    # Data augmentation and normalization for training
    # Just normalization for validation
    if args.model_name == 'resnet101':
        MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
    }

    logger.info("Initializing Datasets and Dataloaders...")

    train_set = LoadSamples(csv_file=args.train_path + '/cas_annotations.csv',
                            root_dir=args.train_path + '/fake/',
                            transform=data_transforms['train'])

    val_set = LoadSamples(csv_file=args.val_path + '/cas_annotations.csv',
                          root_dir=args.val_path + '/real/',
                          transform=data_transforms['val'])

    test_set = LoadSamples(csv_file=args.test_path + '/cas_annotations.csv',
                           root_dir=args.test_path + '/real/',
                           transform=data_transforms['test'])

    # Create training and validation datasets
    image_datasets = {'train': train_set, 'val': val_set, 'test': test_set}

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                       batch_size=args.batch_size,
                                       shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    logger.info("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                logger.info("\t" + name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                logger.info("\t" + name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, logger,
                                 num_epochs=args.total_epoch,
                                 is_inception=(args.model_name == "inception"))

    elapsed = time.time() - start_time
    elapsed = str(datetime.timedelta(seconds=elapsed))
    logger.info("Time Elapsed: [{}]".format(elapsed))


if __name__ == "__main__":
    commit_obj = get_recent_commit()
    current_time = time.strftime("_%d-%m-%Y_%H_%M_%S", time.localtime())
    path = commit_obj + current_time + "_eval_cas_objects"

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='',
                        help='training dataset')
    parser.add_argument('--val_path', type=str, default='',
                        help='validation dataset')
    parser.add_argument('--test_path', type=str, default='',
                        help='test dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='mini-batch size of training data. Default: 128')
    parser.add_argument('--total_epoch', type=int, default=200,
                        help='number of total training epoch')
    parser.add_argument('--model_name', type=str, default='resnet101',
                        help='name of the model to use: "resnet101", "resnet101-cifar"')
    parser.add_argument('--num_classes', type=int, default=178,
                        help='number of classes')
    parser.add_argument('--use_pretrained', default=True, action='store_false',
                        help='if True fine-tune ImageNet pre-trained model')
    parser.add_argument('--feature_extract', default=True, action='store_false',
                        help='if True only update the reshaped layer params')
    parser.add_argument('--out_path', type=str,
                        default='./outputs/' + path,
                        help='path to output files')
    parser.add_argument('--message', type=str, default='CAS Objects Evaluation on Visual Genome',
                        help='Print message in log')
    args = parser.parse_args()

    main(args)


