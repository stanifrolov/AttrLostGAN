import argparse
import datetime
import subprocess
import time
import warnings
import os

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from data.dmnist import DialogMNISTDataset
from model.rcnn_discriminator_dmnist import CombineDiscriminator128
from model.resnet_generator_dmnist import ResnetGenerator128
from model.sync_batchnorm import DataParallelWithCallback
from utils.logger import setup_logger

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def get_dataset():
    global data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data = DialogMNISTDataset(csv_file=args.dataset_path + 'MNIST_Dialog/train_annotations.csv',
                              root_dir=args.dataset_path + 'MNIST_Dialog/train_images/',
                              transform=transform)
    return data


def get_recent_commit():
    return str(subprocess.check_output(["git", "describe", "--always"]).strip().decode())


def main(args):
    # parameters
    z_dim = 128
    lamb_obj = args.lamb_obj
    lamb_img = args.lamb_img
    lamb_dc = args.lamb_dc
    lamb_bgc = args.lamb_bgc
    lamb_ds = args.lamb_ds

    num_classes = 11  # 10 total digits + 1 null class
    num_obj = 8
    num_dcolor = 5
    num_bgcolor = 4
    num_dstyle = 3

    # data loader
    train_data = get_dataset()

    dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        drop_last=True, shuffle=True, num_workers=8)

    # Load model
    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3,
                              num_dcolor=num_dcolor, num_bgcolor=num_bgcolor, num_dstyle=num_dstyle).cuda()
    netD = CombineDiscriminator128(num_classes=num_classes,
                                   num_dcolor=num_dcolor, num_bgcolor=num_bgcolor, num_dstyle=num_dstyle).cuda()

    parallel = True
    if parallel:
        netG = DataParallelWithCallback(netG)
        netD = torch.nn.DataParallel(netD)

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
    current_time = time.strftime("%H_%M_%dd_%mm", time.localtime())

    # make dirs
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.makedirs(os.path.join(args.out_path, 'model/'))
    writer = SummaryWriter(os.path.join(args.out_path, 'log'))

    logger = setup_logger("AttrLostGAN", args.out_path, 0,
                          filename=current_time + '_log.txt')
    logger.info('Commit Tag: ' + commit_obj)
    logger.info(args.message)
    logger.info(args)
    logger.info(netG)
    logger.info(netD)

    total_steps = len(dataloader)
    start_time = time.time()
    for epoch in range(args.total_epoch):
        netG.train()
        netD.train()
        print('epoch ', epoch)

        for idx, data in enumerate(dataloader):
            real_images, label, bbox, dcolor, bgcolor, dstyle = data
            real_images, label, bbox = real_images.cuda(), label.float(), bbox.float()
            dcolor, bgcolor, dstyle = dcolor.float(), bgcolor.float(), dstyle.float()

            # update D network
            netD.zero_grad()
            real_images, label = real_images.float().cuda(), label.long().cuda()
            dcolor, bgcolor, dstyle = dcolor.long().cuda(), bgcolor.long().cuda(), dstyle.long().cuda()

            d_out_real, d_out_robj, d_out_rdcolor, d_out_rbgcolor, d_out_rdstyle = netD(real_images, bbox, label,
                                                                                        dcolor, bgcolor, dstyle)

            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
            d_loss_rdcolor = torch.nn.ReLU()(1.0 - d_out_rdcolor).mean()
            d_loss_rbgcolor = torch.nn.ReLU()(1.0 - d_out_rbgcolor).mean()
            d_loss_rdstyle = torch.nn.ReLU()(1.0 - d_out_rdstyle).mean()

            z = torch.randn(real_images.size(0), num_obj, z_dim).cuda()
            fake_images = netG(z, bbox, y=label.squeeze(dim=-1), dc=dcolor, bgc=bgcolor, ds=dstyle)
            d_out_fake, d_out_fobj, d_out_fdcolor, d_out_fbgcolor, d_out_fdstyle = netD(fake_images.detach(),
                                                                                        bbox, label,
                                                                                        dcolor, bgcolor, dstyle)

            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()
            d_loss_fdcolor = torch.nn.ReLU()(1.0 + d_out_fdcolor).mean()
            d_loss_fbgcolor = torch.nn.ReLU()(1.0 + d_out_fbgcolor).mean()
            d_loss_fdstyle = torch.nn.ReLU()(1.0 + d_out_fdstyle).mean()

            d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake) + \
                     lamb_dc * (d_loss_rdcolor + d_loss_fdcolor) + lamb_bgc * (d_loss_rbgcolor + d_loss_fbgcolor) + \
                     lamb_ds * (d_loss_rdstyle + d_loss_fdstyle)
            d_loss.backward()
            d_optimizer.step()

            # update G network
            if (idx % 1) == 0:
                netG.zero_grad()
                g_out_fake, g_out_obj, g_out_fdcolor, g_out_fbgcolor, g_out_fdstyle = netD(fake_images, bbox, label,
                                                                                           dcolor, bgcolor,
                                                                                           dstyle)
                g_loss_fake = - g_out_fake.mean()
                g_loss_obj = - g_out_obj.mean()
                g_loss_fdcolor = - g_out_fdcolor.mean()
                g_loss_fbgcolor = - g_out_fbgcolor.mean()
                g_loss_fdstyle = - g_out_fdstyle.mean()
                g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img + \
                         g_loss_fdcolor * lamb_dc + \
                         g_loss_fbgcolor * lamb_bgc + \
                         g_loss_fdstyle * lamb_ds
                g_loss.backward()
                g_optimizer.step()

            if (idx + 1) % 100 == 0:
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
                logger.info(args)

                writer.add_image("real images", make_grid(real_images.cpu().data * 0.5 + 0.5, nrow=4),
                                 epoch * total_steps + idx + 1)
                writer.add_image("fake images", make_grid(fake_images.cpu().data * 0.5 + 0.5, nrow=4),
                                 epoch * total_steps + idx + 1)

                writer.add_scalar('DLoss', d_loss, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/real_images', d_loss_real, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/fake_images', d_loss_fake, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/real_objects', d_loss_robj, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/real_dcolor', d_loss_rdcolor, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/real_bgcolor', d_loss_rbgcolor, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/real_dstyle', d_loss_rdstyle, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/fake_objects', d_loss_fobj, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/fake_dcolor', d_loss_fdcolor, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/fake_bgcolor', d_loss_fbgcolor, epoch * total_steps + idx + 1)
                writer.add_scalar('DLoss/fake_dstyle', d_loss_fdstyle, epoch * total_steps + idx + 1)

                writer.add_scalar('GLoss', g_loss.item(), epoch * total_steps + idx + 1)
                writer.add_scalar('GLoss/fake_images', g_loss_fake.item(), epoch * total_steps + idx + 1)
                writer.add_scalar('GLoss/fake_objects', g_loss_obj.item(), epoch * total_steps + idx + 1)
                writer.add_scalar('GLoss/fake_dcolor', g_loss_fdcolor.item(), epoch * total_steps + idx + 1)
                writer.add_scalar('GLoss/fake_bgcolor', g_loss_fbgcolor.item(), epoch * total_steps + idx + 1)
                writer.add_scalar('GLoss/fake_dstyle', g_loss_fdstyle.item(), epoch * total_steps + idx + 1)

        # save model
        if (epoch + 1) % 5 == 0:
            torch.save(netG.state_dict(),
                       os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch + 1)))


if __name__ == "__main__":
    commit_obj = get_recent_commit()
    current_time = time.strftime("_%H_%M_%dd_%mm", time.localtime())
    path = commit_obj + current_time

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default='./datasets/',
                        help='path to mnist dialog dataset')
    parser.add_argument('--out_path', type=str,
                        default='/outputs/' + path,
                        help='path to output files')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='mini-batch size of training data. Default: 128')
    parser.add_argument('--total_epoch', type=int, default=200,
                        help='number of total training epoch')
    parser.add_argument('--d_lr', type=float, default=0.0001,
                        help='learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for generator')
    parser.add_argument('--lamb_obj', type=float, default=1.0,
                        help='Loss weight for objects')
    parser.add_argument('--lamb_img', type=float, default=0.1,
                        help='Loss weight for objects')
    parser.add_argument('--lamb_dc', type=float, default=1.0,
                        help='Loss weight for dcolor attribute')
    parser.add_argument('--lamb_bgc', type=float, default=1.0,
                        help='Loss weight for bgcolor attribute')
    parser.add_argument('--lamb_ds', type=float, default=1.0,
                        help='Loss weight for dstyle attribute')
    parser.add_argument('--message', type=str, default='MNIST Dialog',
                        help='Print a message in log')
    args = parser.parse_args()
    main(args)
