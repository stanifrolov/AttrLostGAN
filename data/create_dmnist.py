import argparse

import random
import operator as op
import os
import struct
from scipy.ndimage.morphology import grey_dilation
import pandas as pd
from pandas.core.common import flatten

import torch
from PIL import Image
from shapely.geometry import box
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from data.dmnist_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--np_rand_seed', type=int, default=1,
                    help='random seed for DM generation')
parser.add_argument('--num_iter', type=int, default=40000,
                    help='n iterations to save n images')
args = parser.parse_args()
print(args)

np.random.seed(args.np_rand_seed)

mnist_dir = './datasets/'
samples_dir = './datasets/MNISTDataset/'


def find_positions_for_digits(new_box, box_list):
    overlap_list = []
    temp_box_list = []
    for bb in box_list:
        interArea = new_box.intersection(bb).area
        new_box_area = new_box.area
        percent_overlap = interArea / new_box_area
        overlap_list.append(percent_overlap)

    # ensure no box overlaps another by more than 40%
    if all(overlap <= 0.4 for overlap in overlap_list):
        # append new box to a temp list since it could intersect with more than one box
        temp_box_list = list(box_list)
        temp_box_list.append(new_box)
        temp_box_list = pd.Series(temp_box_list)

        df = pd.DataFrame()
        for i in range(len(temp_box_list)):
            df[i] = temp_box_list.map(lambda x: x.intersects(temp_box_list[i]))
        # In all columns of df:
        # sum == 1 if a box doesn't intersect any box (Always intersects itself)
        # sum == 2 if a box intersects one other bbox
        # sum >= 3 if a box intersects 2 or more bboxes
        if all(j <= 2 for j in list(df.sum())):
            return True
    return False


def make_bbox(x, y, w=28, h=28):
    bbox = np.array([float(x), float(y), float(w), float(h)])
    return bbox


class MNISTDataset(Dataset):
    def __init__(self, path='.', dataset='training', transform=None):
        self.target_transform = None
        self.path = path
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        len = 60000
        return len

    def __getitem__(self, index):
        label = []
        color = []
        bgcolor = []
        style = []
        images = []
        x_width, y_height = [], []
        boxes = list()

        canvas = Image.new('RGB', (128, 128), 0)

        for rand_imgs in range(random.randint(3, 8)):
            img_data = generateGridImg(1)
            for i in range(len(img_data)):
                for j in range(len(img_data)):
                    label.append(img_data[i][j]['number'])
                    color.append(img_data[i][j]['color'])
                    bgcolor.append(img_data[i][j]['bgcolor'])
                    style.append(img_data[i][j]['style'])

            self.numPools = self.getNumPools()
            self.colorMap = {'blue': [49, 89, 191], 'red': [186, 29, 18], 'green': [62, 140, 33],
                             'violet': [130, 58, 156],
                             'brown': [119, 57, 19], 'white': [255, 255, 255], 'cyan': [71, 255, 253],
                             'salmon': [255, 173, 148],
                             'yellow': [252, 251, 100], 'silver': [204, 204, 204]}

            for k in self.colorMap:
                self.colorMap[k] = np.array(self.colorMap[k], dtype='float32').reshape((1, 1, 3)) / 255

            img = self.realizeGrid(img_data, dataset=self.dataset)

            pil_img = Image.fromarray(np.uint8(img * 255))
            x, y = random.randint(15, 28), random.randint(15, 28) # random resize
            pil_img = pil_img.resize((x, y), Image.ANTIALIAS)
            x_width.append(x)
            y_height.append(y)
            images.append(pil_img)

        grid_size = len(images)

        x, y = [], []
        ready = False
        position_found = False
        num_positions = 0
        position_find_attempts = 0

        while not ready:
            for i, im in enumerate(images):
                while position_find_attempts < 100:
                    cx = random.choice(range(0, 127 - x_width[i]))
                    cy = random.choice(range(0, 127 - y_height[i]))
                    x.append(cx)
                    y.append(cy)
                    new_box = make_bbox(cx, cy, x_width[i], y_height[i])
                    minx, miny, maxx, maxy = new_box[0], new_box[1], new_box[0] + new_box[2], new_box[1] + new_box[3]
                    new_box = box(minx, miny, maxx, maxy)

                    if i == 0:
                        position_found = True
                    else:
                        position_found = find_positions_for_digits(new_box, boxes)

                    if position_find_attempts == 99:  # accept last position if no pos found after 100 attempts
                        position_found = True

                    if position_found:
                        break
                    position_find_attempts += 1

                if position_found:
                    boxes.append(new_box)
                    canvas.paste(im, (cx, cy))
                    num_positions += 1

                if num_positions == grid_size:
                    ready = True
        for i in range(len(boxes)):
            boxes[i] = [boxes[i].bounds[0], boxes[i].bounds[1], boxes[i].bounds[2] - boxes[i].bounds[0],
                        boxes[i].bounds[3] - boxes[i].bounds[1]]

        if self.transform is not None:
            canvas = self.transform(canvas)

        if self.target_transform is not None:
            label = np.array(label)
            label = self.transform(label)
        label = torch.tensor(label, dtype=torch.int8)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        return canvas, label, boxes, color, bgcolor, style

    def read(self, path='.', dataset='training'):
        if dataset is 'training':
            fname_img = os.path.join(path, 'train-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
        elif dataset is 'testing':
            fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

        get_img = lambda idx: (lbl[idx], img[idx])

        # create an iterator which returns each image in turn
        for i in range(len(lbl)):
            yield get_img(i)

    def getNumPools(self):
        pools = {}
        for dataset in ['training', 'testing']:
            mnist = self.read(path=mnist_dir + '/MNISTDataset/raw')
            mnist = sorted(mnist, key=op.itemgetter(0))
            mnist = [(x[0], x[1].astype('float32') / 255) for x in mnist]

            numPools = []
            for i in range(9):  # for mnist digits
                count = 0
                for j in range(len(mnist)):
                    if mnist[j][0] != i:
                        break
                    count += 1
                numPools.append(mnist[:count])
                mnist = mnist[count:]
            numPools.append(mnist)

            pools[dataset] = numPools
        return pools

    def realizeSingleNumber(self, info, size=28, dataset='training'):
        palette = np.ones((size, size, 3), dtype='float32') * self.colorMap[info['bgcolor']]

        num_sample_idx = np.random.randint(len(self.numPools[dataset][info['number']]))
        num_sample = self.numPools[dataset][info['number']][num_sample_idx][1]

        if info['style'] == 'stroke':
            mask = grey_dilation(num_sample, (3, 3)).reshape((28, 28, 1))
            palette = palette * (1 - mask)

        mask = num_sample.reshape((size, size, 1))
        palette = palette * (1 - mask) + (mask * self.colorMap[info['color']]) * mask

        return palette

    def realizeGrid(self, gridImg, size=28, dataset='training'):
        "@returns: Grid of images based on json image data provided and MNIST dataset."
        img = np.zeros((size * len(gridImg), size * len(gridImg), 3))
        for i in range(len(gridImg)):
            for j in range(len(gridImg[i])):
                img[i * size:(i + 1) * size, j * size:(j + 1) * size, :] = self.realizeSingleNumber(gridImg[i][j],
                                                                                                    size=size,
                                                                                                    dataset=dataset)
        return img


transform = transforms.Compose([
    transforms.ToTensor()
])

train_loader = DataLoader(
    MNISTDataset(mnist_dir, transform=transform),
    batch_size=1, shuffle=True)


df = pd.DataFrame(columns={'name', 'labels', 'bbox', 'color', 'bgcolor', 'style'})
for i, data in enumerate(train_loader):
    img, labels, bboxes, color, bgcolor, style = data
    print('Saving Image: ' + '{}.png'.format(i))
    save_image(img, samples_dir + '/images/{idx}.png'.format(idx=i), nrow=1)
    df.loc[i] = {'name': '{idx}.png'.format(idx=i),
                 'labels': list(flatten(np.asarray(labels))),
                 'bbox': np.array2string(np.asarray(bboxes), separator=', '),
                 'color': list(flatten(color)),
                 'bgcolor': list(flatten(bgcolor)),
                 'style': list(flatten(style))
                 }
    df = df[['name', 'labels', 'bbox', 'color', 'bgcolor', 'style']]
    df.to_csv(samples_dir + '/annotations.csv', index=False)
    if i == args.num_iter:
     break
print('DONE')
