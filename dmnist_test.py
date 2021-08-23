import argparse
from collections import OrderedDict
import os

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from data.dmnist import DialogMNISTDataset
from model.resnet_generator_dmnist import ResnetGenerator128
from utils.util_v2 import truncted_random


def drawrect(drawcontext, xy, outline=None, width=1):
    x1, y1, x2, y2 = xy
    x2 = x1 + x2
    y2 = y1 + y2
    x1 += width
    y1 += width
    x2 -= width
    y2 -= width
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    drawcontext.line(points, width=width, fill=outline)


def optimal_position(current_bboxes, bbox_to_add, text_size, line_width):
    x1, y1, x2, y2 = bbox_to_add
    t_w, t_h = text_size
    start_x = x1 - t_w
    start_y = y1 - t_h
    end_x = max(x2 + t_w, 256)
    end_y = max(y2 + t_h, 256)
    cur_x = start_x
    cur_y = start_y

    def get_candidate_boxes():
        nonlocal cur_x
        nonlocal cur_y
        while cur_x < end_x:
            while cur_y < end_y:
                if cur_x > 0 and cur_y > 0 and cur_x + t_w < 256 and cur_y + t_h < 256:
                    candidate = [cur_x, cur_y, cur_x + t_w, cur_y + t_h]
                    yield candidate
                cur_y += t_h // 3
            cur_x += t_w // 3
            cur_y = start_y

    for caption_bbox in get_candidate_boxes():
        valid = True
        for cur_bbox in current_bboxes:
            if box_has_intersect(caption_bbox, cur_bbox):
                valid = False
                break
        if valid:
            return caption_bbox
    if len(current_bboxes) > 0:
        print("no solution found")
    return None


def box_has_intersect(bbox1, bbox2):
    return bb_intersection_over_union(bbox1, bbox2) > 1e-2


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    eps = 1e-5
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + eps) * max(0, yB - yA + eps)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + eps) * (boxA[3] - boxA[1] + eps)
    boxBArea = (boxB[2] - boxB[0] + eps) * (boxB[3] - boxB[1] + eps)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


color_names = ['red', 'blue', 'purple', 'green', 'olive', 'fuchsia', 'gold', 'aqua', 'teal', 'orange']
font = ImageFont.load_default()
#font = ImageFont.truetype(<path_to_arial.ttf>, 8)

dcolor_list = ['blue', 'red', 'green', 'violet', 'brown']
bgcolor_list = ['white', 'cyan', 'salmon', 'yellow', 'silver']
dstyle_list = ['flat', 'stroke']


def draw_layout(bboxes, label, dcolor, bgcolor, dstyle, attr_type, draw_caption, default_size=128, num_objs=8):
    line_width = 2
    img_canvas = Image.new('RGBA', (default_size, default_size))
    draw_cxt = ImageDraw.Draw(img_canvas)
    drawrect(draw_cxt, [0, 0, 1 * default_size, 1 * default_size], outline=color_names[0], width=line_width)
    bboxes = bboxes.squeeze(0)
    label = label.squeeze(0)
    dcolor = dcolor.squeeze(0)
    bgcolor = bgcolor.squeeze(0)
    dstyle = dstyle.squeeze(0)

    x = 0
    for i in range(num_objs):
        if bboxes[i][2] == 0.5:
            pass
        else:
            x += 1
    num_objs = x

    current_caption_boxes = []
    for i in range(num_objs):
        bbox = bboxes[i]
        bbox = [int(x) for x in bbox * default_size]
        if bbox[2] - bbox[0] < default_size * 0.3 and bbox[3] - bbox[1] < default_size * 0.3:
            current_caption_boxes.append(bbox)
        drawrect(draw_cxt, bbox, outline=color_names[i + 1], width=line_width)

    for i in range(num_objs):
        bbox = bboxes[i]
        bbox = [int(x) for x in bbox * default_size]

        text = str(label[i])
        if dcolor is not None:
            if attr_type == 'dcolor':
                text = text + " " + str(dcolor_list[dcolor[i]])
            elif attr_type == 'bgcolor':
                text = text + " " + str(bgcolor_list[bgcolor[i]])
            elif attr_type == 'dstyle':
                text = text + " " + str(dstyle_list[dstyle[i]])
            elif attr_type == 'all':
                text = text + " " + str(dcolor_list[dcolor[i]]) \
                       + ", " + str(bgcolor_list[bgcolor[i]]) \
                       + ", " + str(dstyle_list[dstyle[i]])

        text_size = font.getsize(text)
        text_bbox = Image.new('RGBA', text_size, color_names[i + 1])
        if draw_caption:
            new_caption_bbox = optimal_position(current_caption_boxes,
                                                bbox, text_size, line_width)
            if new_caption_bbox is None:
                continue
            current_caption_boxes.append(new_caption_bbox)
            new_x, new_y, _, _ = list(map(int, new_caption_bbox))
            img_canvas.paste(text_bbox, (new_x, new_y))
            draw_cxt.text((new_x, new_y), text,
                          font=font, fill='white')
    return img_canvas


def get_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = DialogMNISTDataset(csv_file=args.dataset_path + 'MNIST_Dialog/test_annotations.csv',
                                     root_dir=args.dataset_path + 'MNIST_Dialog/test_images/',
                                     transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        drop_last=True, shuffle=False, num_workers=8)

    return dataloader


def crop(image, bbox):
    x, y, w, h = int(bbox[0] * 128), int(bbox[1] * 128), int(bbox[2] * 128), int(bbox[3] * 128)
    image = image[:, y:y + h, x:x + w]
    return image


def main(args):
    num_classes = 11
    num_o = 8
    num_dcolor = 6
    num_bgcolor = 6
    num_dstyle = 3

    dataloader = get_dataloader()

    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3,
                              num_dcolor=num_dcolor, num_bgcolor=num_bgcolor, num_dstyle=num_dstyle).cuda()

    if not os.path.isfile(args.model_path):
        return
    state_dict = torch.load(args.model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.cuda()
    netG.eval()

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
        os.makedirs(args.sample_path + '/fcanvas')
        os.makedirs(args.sample_path + '/real')
        os.makedirs(args.sample_path + '/fake')
        os.makedirs(args.sample_path + '/layout')

    thres = 2.0
    df = pd.DataFrame()
    for idx, data in enumerate(dataloader):
        real_images, label, bbox, dcolor, bgcolor, dstyle = data
        real_images, label = real_images.cuda(), label.long().unsqueeze(-1).cuda()
        dcolor, bgcolor, dstyle = dcolor.long().cuda(), bgcolor.long().cuda(), dstyle.long().cuda()  # to cuda

        z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().cuda()  # [1, 16, 128]
        z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()  # [1, 2048]
        fake_images = netG.forward(z_obj, bbox.cuda(), z_im, label.squeeze(dim=-1),
                                   dcolor, bgcolor, dstyle)

        imgs = fake_images[0].cpu().detach().numpy()
        imgs = imgs.transpose(1, 2, 0) * 0.5 + 0.5  # [128,128,3]
        imgs = imgs * 255

        imageio.imwrite("{save_path}/fcanvas/sample_{idx}.png".format(save_path=args.sample_path, idx=idx),
                        imgs.astype('uint8'))
        layout = draw_layout(bbox.cpu().detach().numpy(),
                             label.cpu().detach().numpy(),
                             dcolor.cpu().detach().numpy(),
                             bgcolor.cpu().detach().numpy(),
                             dstyle.cpu().detach().numpy(),
                             'all',
                             draw_caption=True)
        layout.save("{sample_path}/layout/layout_{idx}.png".format(sample_path=args.sample_path, idx=idx))

        for i in range(num_o):
            if not bbox[0][i][0] == -0.6:  # don't save null classes
                imgs = crop(fake_images.squeeze().cpu().detach().numpy(), bbox[0][i].cpu().detach().numpy())
                rimgs = crop(real_images.squeeze().cpu().detach().numpy(), bbox[0][i].cpu().detach().numpy())
                imgs = imgs.transpose(1, 2, 0) * 0.5 + 0.5
                imgs = imgs * 255
                rimgs = rimgs.transpose(1, 2, 0) * 0.5 + 0.5
                rimgs = rimgs * 255

                imageio.imwrite("{save_path}/{idx}{i}.png".format(save_path=args.sample_path + '/fake', idx=idx, i=i),
                                imgs.astype('uint8'))

                imageio.imwrite("{save_path}/{idx}{i}.png".format(save_path=args.sample_path + '/real', idx=idx, i=i),
                                rimgs.astype('uint8'))

                d = {'name': '{idx}{i}.png'.format(idx=idx, i=i),
                     'labels': np.array2string(np.asarray(label[0][i].cpu()), separator=', '),
                     'bbox': np.array2string(np.asarray(bbox[0][i].cpu()), separator=', '),
                     'dcolor': np.array2string(np.asarray(dcolor[0][i].cpu()), separator=', '),
                     'bgcolor': np.array2string(np.asarray(bgcolor[0][i].cpu()), separator=', '),
                     'dstyle': np.array2string(np.asarray(dstyle[0][i].cpu()), separator=', ')}

                df = df.append(d, ignore_index=True)
                print('Saving image {}{}'.format(idx, i))
    print('Images Saved!')
    df = df[['name', 'labels', 'bbox', 'dcolor', 'bgcolor', 'dstyle']]
    df.to_csv(args.sample_path + '/cas_annotations.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default='./datasets/',
                        help='path to mnist dialog dataset')
    parser.add_argument('--model_path', type=str,
                        help='which epoch to load')
    parser.add_argument('--sample_path', type=str, default='./samples/',
                        help='path to save generated images')
    args = parser.parse_args()
    main(args)
