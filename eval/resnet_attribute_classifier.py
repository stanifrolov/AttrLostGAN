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
from PIL import Image
from utils.logger import setup_logger


print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_recent_commit():
    return str(subprocess.check_output(["git", "describe", "--always"]).strip().decode())


def safe_divide(a, b):
    return torch.stack([torch.as_tensor(torch.true_divide(x, y)) if y else torch.tensor(1.0).cuda() for x, y in zip(a, b)])


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
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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

    val_exact_match_ratio_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_exact_match_ratio = 0.0

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
            running_exact_matches = 0.0
            running_correct_predictions = 0.0
            running_hamming = 0.0
            running_naive_acc = 0.0
            running_acc = 0.0
            running_precision = 0.0
            running_recall = 0.0
            running_f_one = 0.0
            running_balanced_acc = 0.0

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

                    preds = outputs

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)

                predictions = preds.data > 0.5
                targets = labels.data > 0.5
                predicted_correct = torch.sum(predictions & targets, axis=1)

                running_exact_matches += torch.all(predictions == targets, axis=1).sum()
                running_correct_predictions += torch.true_divide(predicted_correct,
                                                                 args.num_classes).sum()

                # how many different bits
                running_hamming += torch.true_divide(torch.sum(predictions != targets, axis=1),
                                                     args.num_classes).sum()

                running_naive_acc += torch.true_divide(torch.sum(predictions == targets, axis=1),
                                                       args.num_classes).sum()

                running_acc += torch.true_divide(predicted_correct,
                                                 torch.sum(predictions | targets, axis=1)).sum()

                # predicted correct labels / total number of predicted labels
                running_precision += safe_divide(predicted_correct,
                                                 torch.sum(predictions, axis=1)).sum()

                # predicted correct labels / total number of actual labels
                running_recall += torch.true_divide(predicted_correct,
                                                    torch.sum(targets, axis=1)).sum()

                # harmonic mean of precision and recall
                running_f_one += torch.true_divide(2 * predicted_correct,
                                                   torch.sum(targets, axis=1) + torch.sum(predictions, axis=1)).sum()

                # balanced accuracy: (TPR (recall) + TNR) / 2
                TN = torch.sum((preds.data <= 0.5) & (labels.data <= 0.5), axis=1)
                TNR = torch.true_divide(TN, torch.sum(labels.data <= 0.5, axis=1))
                running_balanced_acc += torch.true_divide(
                                            torch.true_divide(predicted_correct,
                                                              torch.sum(targets, axis=1)) + TNR, 2).sum()

                # save predictions
                if phase in ['val', 'test']:
                    for i in range(len(labels)):
                        d = {
                            'target': np.asarray(labels[i].data.cpu()),
                            'prediction': np.asarray(preds[i].data.cpu()),
                        }

                        df = df.append(d, ignore_index=True)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_exact_match_ratio = running_exact_matches / len(dataloaders[phase].dataset)
            epoch_correct_predictions = running_correct_predictions / len(dataloaders[phase].dataset)
            epoch_hamming = running_hamming / len(dataloaders[phase].dataset)
            epoch_naive_acc = running_naive_acc / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase].dataset)
            epoch_balanced_acc = running_balanced_acc / len(dataloaders[phase].dataset)
            epoch_precision = running_precision / len(dataloaders[phase].dataset)
            epoch_recall = running_recall / len(dataloaders[phase].dataset)
            epoch_f_one = running_f_one / len(dataloaders[phase].dataset)

            logger.info('{} Loss: {:.4f}, '
                        'Exact-Match-Ratio: {:.4f}, '
                        'Correct Predictions: {:.4f}, '
                        'Hamming Loss ("Wrong Bits"): {:.4f}, '
                        'Naive Acc: {:.4f}, '
                        'Acc: {:.4f}, '
                        'Balanced Acc: {:.4f}, '
                        'Precision: {:.4f}, '
                        'Recall: {:.4f}, '
                        'F1: {:.4f}'
                        .format(phase, epoch_loss,
                                epoch_exact_match_ratio, epoch_correct_predictions, epoch_hamming,
                                epoch_naive_acc, epoch_acc, epoch_balanced_acc,
                                epoch_precision, epoch_recall,
                                epoch_f_one))

            # deep copy the model
            if phase == 'val' and epoch_exact_match_ratio > best_exact_match_ratio:
                best_exact_match_ratio = epoch_exact_match_ratio
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_exact_match_ratio_history.append(epoch_exact_match_ratio)

            if phase in ['val', 'test']:
                df = df[['target', 'prediction']]
                df.to_csv(os.path.join(args.out_path, phase + '_predictions', 'epoch_%d.csv' % (epoch + 1)), index=False)

        torch.save(model.state_dict(),
                   os.path.join(args.out_path, 'model/', 'model_%d.pth' % (epoch + 1)))

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Exact-Match-Ratio: {:4f}'.format(best_exact_match_ratio))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_exact_match_ratio_history


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
        self.annotations['attributes'] = self.annotations['attributes'].apply(literal_eval)

        # indices to crop annotations with attribute information
        self.indices = list(self.annotations[self.annotations['attributes_name'].map(lambda d: len(d)) > 2]['attributes_name'].index)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        img_name = os.path.join(self.root_dir, self.annotations.loc[idx, 'name'])
        image = Image.open(img_name).convert('RGB')

        attributes = self.annotations.loc[idx, 'attributes']
        attributes = torch.tensor(attributes)

        if self.transform is not None:
            image = self.transform(image)

        return image, attributes


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
    #print(model_ft)

    # Data augmentation and normalization for training
    # Just normalization for validation
    if args.model_name == 'resnet101':
        MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        MEAN,  STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

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

    attribute_idx_to_name = ["white", "black", "green", "blue", "red", "brown", "yellow", "small", "large", "wooden",
                             "gray", "silver", "metal", "orange", "grey", "tall", "long", "dark", "pink", "standing",
                             "clear", "white ", "round", "tan", "here", "wood", "glass", "open", "purple", "big",
                             "short", "plastic", "parked", "sitting", "walking", "black ", "striped", "young", "brick",
                             "gold", "empty", "hanging", "old", "on", "blue ", "bright", "concrete", "cloudy", "green ",
                             "colorful", "one", "beige", "bare", "wet", "closed", "square", "light", "stone", "little",
                             "blonde", "shiny", "red ", "dirty", "thin", "smiling", "painted", "flying", "brown ",
                             "thick", "sliced", "playing", "calm", "looking", "distant", "tennis", "part", "leather",
                             "dry", "rectangular", "grassy"]

    # number of crops with attribute labels
    num_crops = 235970
    # in the training set on 80 attributes
    attribute_occurences_per_index = [50571, 30054, 23916, 23700, 14310, 21074, 8221, 6231, 10219,
                                    7280, 7565, 2798, 4121, 3567, 6701, 5940, 4619, 4540,
                                    2731, 4529, 4394, 1473, 2176, 3120, 3144, 2667, 1927,
                                    2077, 1574, 2490, 2579, 1494, 2396, 2906, 2189, 780,
                                    2109, 2177, 2258, 730, 1450, 1000, 1410, 467, 751,
                                    1161, 1016, 2247, 670, 882, 644, 1098, 1525, 1290,
                                    720, 804, 922, 1013, 906, 1475, 702, 368, 979,
                                    670, 1531, 794, 510, 495, 794, 226, 1035, 978,
                                    1296, 515, 143, 587, 523, 795, 456, 820]

    if args.attrlayout2im:
        attribute_occurences_per_index = [1527, 382, 52768, 7506, 967, 370, 2221, 4698, 14875, 1321, 871, 848, 1307,
                                          405, 754, 442, 391, 1449, 2739, 525, 479, 912, 2234, 726, 6139, 21852, 3704,
                                          494, 2909, 482, 977, 1644, 604, 363, 475, 584, 2261, 428, 526, 4530, 627, 436,
                                          4589, 389, 2838, 2118, 753, 2650, 2277, 647, 513, 2152, 1547, 1994, 378, 1042,
                                          356, 599, 7807, 994, 541, 1134, 10609, 6488, 482, 929, 401, 3208, 533, 2442,
                                          8607, 379, 478, 396, 2560, 362, 820, 4272, 726, 460, 2930, 1549, 647, 473,
                                          4799, 552, 1059, 373, 832, 474, 24952, 1196, 815, 514, 24810, 31277, 659, 486,
                                          444, 1566, 2337, 539, 993, 690, 6912, 939]
        num_crops = 246840

    pos_weight = [(num_crops - i) / i for i in attribute_occurences_per_index]
    pos_weight = torch.FloatTensor(pos_weight).cuda()

    # Setup the loss fxn
    if args.no_weighted_loss:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
    path = commit_obj + current_time + "_eval_cas_attributes"

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
    parser.add_argument('--no_weighted_loss', default=False, action='store_true',
                        help='if True, weights are not used in the loss"')
    parser.add_argument('--num_classes', type=int, default=80,
                        help='number of classes')
    parser.add_argument('--use_pretrained', default=True, action='store_false',
                        help='if True fine-tune ImageNet pre-trained model')
    parser.add_argument('--feature_extract', default=True, action='store_false',
                        help='if True only update the reshaped layer params')
    parser.add_argument('--out_path', type=str,
                        default='./outputs/' + path,
                        help='path to output files')
    parser.add_argument('--message', type=str, default='Attr-F1 Evaluation on Visual Genome with Attributes',
                        help='Print message in log')
    parser.add_argument('--attrlayout2im', default=False, action='store_true',
                        help='use 106 attributes and different counts')
    args = parser.parse_args()

    main(args)
