import argparse
import json
import os
import time
import datetime
import subprocess
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import warnings

from data.vg import VgSceneGraphDataset

from utils.logger import setup_logger
from utils.util_v2 import VGGLoss

from model.sync_batchnorm import DataParallelWithCallback

from model.resnet_generator import ResnetGenerator128 as ResnetGenerator128v1
from model.resnet_generator_v2 import ResnetGenerator128 as ResnetGenerator128v2
from model.resnet_generator_v2 import ResnetGenerator256 as ResnetGenerator256v2

from model.rcnn_discriminator import CombineDiscriminator128 as CombineDiscriminator128v1
from model.rcnn_discriminator_v2 import CombineDiscriminator128 as CombineDiscriminator128v2
from model.rcnn_discriminator_v2 import CombineDiscriminator256 as CombineDiscriminator256v2


warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def get_dataset(img_size):
    global data

    with open("./datasets/vocab.json", "r") as read_file:
        vocab = json.load(read_file)

    data = VgSceneGraphDataset(vocab=vocab, h5_path='./datasets/train.h5',
                               image_dir='./datasets/images/',
                               image_size=(img_size, img_size), max_objects=30, left_right_flip=True)
    return data


def get_recent_commit():
    return str(subprocess.check_output(["git", "describe", "--always"]).strip().decode())


def main(args):
    # parameters
    img_size = args.img_size
    z_dim = 128
    lamb_obj = args.lamb_obj
    lamb_img = args.lamb_img
    lamb_attr = args.lamb_attr

    num_classes = 179
    num_obj = 31
    num_attrs = 80

    # data loader
    train_data = get_dataset(img_size)
    print(len(train_data))

    dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        drop_last=True, shuffle=True, num_workers=8)

    # Load model
    if args.version == 1:
        assert args.img_size == 128, "args.img_size should be 128 for version 1"

        netG = ResnetGenerator128v1(num_classes=num_classes, output_dim=3, num_attrs=num_attrs).cuda()
        netD = CombineDiscriminator128v1(num_classes=num_classes, num_attrs=num_attrs).cuda()
    elif args.version == 2:
        if img_size == 128:
            netG = ResnetGenerator128v2(num_classes=num_classes, output_dim=3, num_attrs=num_attrs).cuda()
            netD = CombineDiscriminator128v2(num_classes=num_classes, num_attrs=num_attrs).cuda()
        elif img_size == 256:
            netG = ResnetGenerator256v2(num_classes=num_classes, output_dim=3, num_attrs=num_attrs).cuda()
            netD = CombineDiscriminator256v2(num_classes=num_classes, num_attrs=num_attrs).cuda()
        else:
            assert False, "args.img_size should be 128 or 256"
    else:
        assert False, "args.version should be 1 or 2"

    parallel = True
    if parallel:
        netG = DataParallelWithCallback(netG)
        netD = nn.DataParallel(netD)

    g_lr, d_lr = args.g_lr, args.d_lr
    gen_parameters = []
    for key, value in dict(netG.named_parameters()).items():
        if value.requires_grad:
            if 'mapping' in key:
                gen_parameters += [{'params': [value], 'lr': g_lr * 0.1}]
            else:
                gen_parameters += [{'params': [value], 'lr': g_lr}]

    g_optimizer = torch.optim.Adam(gen_parameters, betas=(0, 0.999))

    dis_parameters = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            dis_parameters += [{'params': [value], 'lr': d_lr}]
    d_optimizer = torch.optim.Adam(dis_parameters, betas=(0, 0.999))

    # get most recent commit
    commit_obj = get_recent_commit()
    current_time = time.strftime("%d-%m-%Y_%H_%M_%dd_%mm", time.localtime())

    # make dirs
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.makedirs(os.path.join(args.out_path, 'model/'))
    writer = SummaryWriter(os.path.join(args.out_path, 'log'))

    logger = setup_logger("AttrLostGAN", args.out_path, 0,
                          filename=current_time + '_log.txt')
    logger.info('Commit Tag: ' + commit_obj)
    logger.info('Time: ' + current_time)
    logger.info(args.message)
    logger.info(args)
    logger.info(netG)
    logger.info(netD)

    total_steps = len(dataloader)

    start_time = time.time()

    if args.version == 2:
        vgg_loss = VGGLoss()
        vgg_loss = nn.DataParallel(vgg_loss)
        l1_loss = nn.DataParallel(nn.L1Loss())

    for epoch in range(args.total_epoch):
        netG.train()
        netD.train()
        print('epoch ', epoch)

        for idx, data in enumerate(dataloader):
            real_images, label, bbox, attrs = data
            real_images, label, bbox, attrs = real_images.cuda(), label.float(), bbox.float(), attrs.float()

            # update D network
            netD.zero_grad()
            real_images, label, attrs = real_images.float().cuda(), label.long().cuda(), attrs.cuda()
            d_out_real, d_out_robj, d_out_rattrs = netD(real_images, bbox, label, attrs)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
            d_loss_rattrs = torch.nn.ReLU()(1.0 - d_out_rattrs).mean()

            z = torch.randn(real_images.size(0), num_obj, z_dim).cuda()
            fake_images = netG(z, bbox, y=label.squeeze(dim=-1), attrs=attrs)
            d_out_fake, d_out_fobj, d_out_fattrs = netD(fake_images.detach(), bbox, label, attrs)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()
            d_loss_fattrs = torch.nn.ReLU()(1.0 + d_out_fattrs).mean()

            d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + \
                     lamb_img * (d_loss_real + d_loss_fake) + \
                     lamb_attr * (d_loss_rattrs + d_loss_fattrs)

            d_loss.backward()
            d_optimizer.step()

            # update G network
            if (idx % 1) == 0:
                netG.zero_grad()
                g_out_fake, g_out_obj, g_out_fattrs = netD(fake_images, bbox, label,attrs)
                g_loss_fake = - g_out_fake.mean()
                g_loss_obj = - g_out_obj.mean()
                g_loss_fattrs = - g_out_fattrs.mean()

                if args.version == 2:
                    pixel_loss = l1_loss(fake_images, real_images).mean()
                    feat_loss = vgg_loss(fake_images, real_images).mean()

                g_loss = g_loss_obj * lamb_obj + \
                         g_loss_fake * lamb_img + \
                         g_loss_fattrs * lamb_attr

                if args.version == 2:
                         g_loss += pixel_loss + feat_loss

                g_loss.backward()
                g_optimizer.step()

            if (idx + 1) % 100 == 0:
                print('SAVING TENSORBOARD VISUALIZATIONS')
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                logger.info("Time Elapsed: [{}]".format(elapsed))
                logger.info("Epoch[{}/{}], Step[{}/{}], "
                            "d_out_real: {:.4f}, d_out_fake: {:.4f}, g_out_fake: {:.4f} ".format(epoch + 1,
                                                                                                 args.total_epoch,
                                                                                                 idx + 1,
                                                                                                 total_steps,
                                                                                                 d_loss_real.item(),
                                                                                                 d_loss_fake.item(),
                                                                                                 g_loss_fake.item()))
                logger.info("d_obj_real: {:.4f}, d_obj_fake: {:.4f}, g_obj_fake: {:.4f} ".format(
                    d_loss_robj.item(),
                    d_loss_fobj.item(),
                    g_loss_obj.item()))

                if args.version == 2:
                    logger.info("pixel_loss: {:.4f}, feat_loss: {:.4f}".format(pixel_loss.item(), feat_loss.item()))

                logger.info(args)

                writer.add_image("real images", make_grid(real_images.cpu().data * 0.5 + 0.5, nrow=4),
                                 epoch * total_steps + idx + 1)
                writer.add_image("fake images", make_grid(fake_images.cpu().data * 0.5 + 0.5, nrow=4),
                                 epoch * total_steps + idx + 1)

                writer.add_scalar('DLoss', d_loss, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/real_images', d_loss_real, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/fake_images', d_loss_fake, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/real_objects', d_loss_robj, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/real_attrs', d_loss_rattrs, epoch * total_steps + idx + 1)

                writer.add_scalar('DLoss/fake_objects', d_loss_fobj, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/fake_attrs', d_loss_fattrs, epoch * total_steps + idx + 1)

                writer.add_scalar('GLoss', g_loss.item(), epoch * total_steps + idx + 1)
                writer.add_scalar('GLoss/fake_images', g_loss_fake.item(), epoch * total_steps + idx + 1)
                writer.add_scalar('GLoss/fake_objects', g_loss_obj.item(), epoch * total_steps + idx + 1)
                writer.add_scalar('GLoss/fake_attrs', g_loss_fattrs.item(), epoch * total_steps + idx + 1)

        # save model
        if (epoch + 1) % 5 == 0:
            torch.save(netG.state_dict(),
                       os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch + 1)))


if __name__ == "__main__":
    commit_obj = get_recent_commit()
    current_time = time.strftime("_%H_%M_%dd_%mm", time.localtime())
    path = commit_obj + current_time

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=1,
                        help='model version 1 or 2, default is 1')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size. Default: 128')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='mini-batch size of training data. Default: 128')
    parser.add_argument('--total_epoch', type=int, default=200,
                        help='number of total training epoch')
    parser.add_argument('--d_lr', type=float, default=0.0001,
                        help='learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for generator')
    parser.add_argument('--out_path', type=str,
                        default='./outputs/' + path,
                        help='path to output files')
    parser.add_argument('--lamb_obj', type=float, default=1.0,
                        help='Loss weight for objects')
    parser.add_argument('--lamb_img', type=float, default=0.1,
                        help='Loss weight for objects')
    parser.add_argument('--lamb_attr', type=float, default=1.0,
                        help='Loss weight for attributes')
    parser.add_argument('--message', type=str, default='Visual Genome with Attributes',
                        help='Print message in log')
    args = parser.parse_args()
    main(args)