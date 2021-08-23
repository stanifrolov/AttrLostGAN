import argparse
import glob
import json
import os
import time
from collections import OrderedDict

import imageio
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from data.vg import VgSceneGraphDataset

from utils.util_v2 import truncted_random

from model.resnet_generator import ResnetGenerator128 as ResnetGenerator128v1
from model.resnet_generator_v2 import ResnetGenerator128 as ResnetGenerator128v2
from model.resnet_generator_v2 import ResnetGenerator256 as ResnetGenerator256v2

start_time = time.time()

with open("./datasets/vocab.json", "r") as read_file:
    vocab = json.load(read_file)


def get_dataloader(img_size=128):
    dataset = VgSceneGraphDataset(vocab=vocab, h5_path='./datasets/' + args.split + '.h5',
                                  image_dir='./datasets/images/',
                                  image_size=(img_size, img_size), max_objects=30, left_right_flip=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             drop_last=False, shuffle=False,
                                             num_workers=8)

    return dataloader


def crop(image, bbox):
    x, y, w, h = [int(bbox[i] * args.img_size) for i in range(4)]
    image = image[:, y:y + h, x:x + w]
    return image


def main(args):
    num_classes = 179
    num_o = 31
    num_attrs = 80

    dataloader = get_dataloader(args.img_size)

    if args.version == 1:
        assert args.img_size == 128, "args.img_size should be 128 for version 1"

        netG = ResnetGenerator128v1(num_classes=num_classes, output_dim=3, num_attrs=num_attrs).cuda()
    elif args.version == 2:
        if args.img_size == 128:
            netG = ResnetGenerator128v2(num_classes=num_classes, output_dim=3, num_attrs=num_attrs).cuda()
        elif args.img_size == 256:
            netG = ResnetGenerator256v2(num_classes=num_classes, output_dim=3, num_attrs=num_attrs).cuda()
        else:
            assert False, "args.img_size should be 128 or 256"
    else:
        assert False, "args.version should be 1 or 2"

    snapshots = glob.glob(args.model_path + "G_200.pth")
    snapshots.sort(key=os.path.getmtime)

    current_date_and_time = time.strftime("%d-%m-%Y_%H_%M_%dd_%mm/", time.localtime())

    for snapshot in snapshots:
        state_dict = torch.load(snapshot)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove `module.`nvidia
            new_state_dict[name] = v

        model_dict = netG.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        netG.load_state_dict(model_dict)

        netG.cuda()
        netG.eval()

        sample_path = args.sample_path \
                      + current_date_and_time \
                      + args.split \
                      + args.model_path.split('AttrLostGAN')[1].replace("/", "_") + '/' \
                      + "_" + args.version \
                      + "_" + args.img_size \
                      + snapshot.split("model/")[1].split(".pth")[0]

        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
            os.makedirs(sample_path + '/fcanvas')
            os.makedirs(sample_path + '/rcanvas')
            os.makedirs(sample_path + '/real')
            os.makedirs(sample_path + '/fake')

        thres = 2.0
        df = pd.DataFrame()
        np.random.seed(1000)

        for dataset_run in range(1):
            for idx, data in enumerate(dataloader):
                real_images, label, bbox, attrs = data
                real_images, label, attrs = real_images.cuda(), label.long().unsqueeze(-1).cuda(), attrs.float().cuda()

                z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().cuda()
                z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()

                fake_images = netG.forward(z_obj, bbox.cuda(), z_im, label.squeeze(dim=-1), attrs)

                imgs = fake_images[0].cpu().detach().numpy()
                imgs = imgs.transpose(1, 2, 0) * 0.5 + 0.5
                imgs = imgs * 255

                rimgs = real_images[0].cpu().detach().numpy()
                rimgs = rimgs.transpose(1, 2, 0) * 0.5 + 0.5
                rimgs = rimgs * 255

                imageio.imwrite("{save_path}/fcanvas/sample_{dataset_run}_{idx}.png"
                                .format(save_path=sample_path,
                                        dataset_run=dataset_run,
                                        idx=f'{idx:06}'),
                                imgs.astype('uint8'))

                imageio.imwrite("{save_path}/rcanvas/rsample_{dataset_run}_{idx}.png"
                                .format(save_path=sample_path,
                                        dataset_run=dataset_run,
                                        idx=f'{idx:06}'),
                                rimgs.astype('uint8'))

                if args.save_crops:
                    for i in range(num_o):
                        if not bbox[0][i][0] == -0.6:  # don't save null classes
                            imgs = crop(fake_images.squeeze().cpu().detach().numpy(), bbox[0][i].cpu().detach().numpy())
                            rimgs = crop(real_images.squeeze().cpu().detach().numpy(),bbox[0][i].cpu().detach().numpy())

                            imgs = imgs.transpose(1, 2, 0) * 0.5 + 0.5
                            imgs = imgs * 255

                            rimgs = rimgs.transpose(1, 2, 0) * 0.5 + 0.5
                            rimgs = rimgs * 255

                            imageio.imwrite("{save_path}/{dataset_run}_{idx}_{i}.png"
                                            .format(save_path=sample_path + '/fake',
                                                    dataset_run=dataset_run,
                                                    idx=f'{idx:06}',
                                                    i=f'{i:02}'),
                                            imgs.astype('uint8'))

                            imageio.imwrite("{save_path}/{dataset_run}_{idx}_{i}.png"
                                            .format(save_path=sample_path + '/real',
                                                    dataset_run=dataset_run,
                                                    idx=f'{idx:06}',
                                                    i=f'{i:02}'),
                                            rimgs.astype('uint8'))

                            d = {'name': '{dataset_run}_{idx}_{i}.png'.format(dataset_run=dataset_run,idx=f'{idx:06}', i=f'{i:02}'),
                                 'label': np.array2string(np.asarray(label[0][i].cpu()), separator=', '),
                                 'label_name': np.array2string(
                                     np.asarray(vocab['object_idx_to_name'][label[0][i].cpu()]), separator=', '),
                                 'bbox': np.array2string(np.asarray(bbox[0][i].cpu()), separator=', '),
                                 'attributes_name': np.array2string(np.asarray(
                                     [vocab['attribute_idx_to_name'][j] for j in
                                      range(len(np.asarray(attrs[0][i].cpu()))) if attrs[0][i][j] != 0]),
                                      separator=', '),
                                 'attributes': np.array2string(np.asarray(attrs[0][i].cpu()), separator=', '),
                                 }
                            df = df.append(d, ignore_index=True)

        print('Images Saved!')
        df = df[['name', 'label', 'label_name', 'bbox', 'attributes_name', 'attributes']]
        df.to_csv(sample_path + '/cas_annotations.csv', index=False)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=1,
                        help='model version 1 or 2, default is 1')
    parser.add_argument('--img_size', type=int, default=128,
                        help='image size 128 or 256, default is 128')
    parser.add_argument('--split', type=str, default='val',
                        help='dataset split "train", "val", or "test"')
    parser.add_argument('--save_crops', type=bool, default=True,
                        help='whether to save crops based on bounding boxes or not')
    parser.add_argument('--model_path', type=str,
                        help='which model to load')
    parser.add_argument('--sample_path', type=str, default='./samples/',
                        help='path to save generated images')
    args = parser.parse_args()
    main(args)
