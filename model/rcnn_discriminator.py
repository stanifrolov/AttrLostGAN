import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.roi_align import RoIAlign as ROIAlign
from utils.util_v2 import *
from utils.bilinear import *


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


class ResnetDiscriminator128(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64, num_attrs=None):
        super(ResnetDiscriminator128, self).__init__()
        self.num_classes = num_classes
        self.num_dcolor = num_attrs

        self.block1 = OptimizedBlock(3, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block6 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.roi_align_s = ROIAlign((8, 8), 1.0 / 4.0, int(0))
        self.roi_align_l = ROIAlign((8, 8), 1.0 / 8.0, int(0))

        self.block_obj3 = ResBlock(ch * 2, ch * 4, downsample=False)
        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))

        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16, padding_idx=0))
        self.attribute_features = nn.utils.spectral_norm(nn.Linear(num_attrs + 1, ch * 16))

    def forward(self, x, y=None, bbox=None, attrs=None):
        b = bbox.size(0)
        # 128x128
        x = self.block1(x)
        # 64x64
        x1 = self.block2(x)
        # 32x32
        x2 = self.block3(x1)
        # 16x16
        x = self.block4(x2)
        # 8x8
        x = self.block5(x)
        # 4x4
        x = self.block6(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l7(x)

        # seperate obj path
        s_idx = (bbox[:, 3] < 64) * (bbox[:, 4] < 64)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]
        y_l, y_s = y[~s_idx], y[s_idx]
        attrs_l, attrs_s = attrs[~s_idx], attrs[s_idx]

        obj_feat_s = self.block_obj3(x1)
        obj_feat_s = self.block_obj4(obj_feat_s)
        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

        obj_feat_l = self.block_obj4(x2)
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)

        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        attrs = torch.cat([attrs_l, attrs_s], dim=0)

        obj_feat = self.block_obj5(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))

        out_obj_l = self.l_obj(obj_feat)
        out_obj = out_obj_l + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        out_attrs = out_obj_l + torch.sum(self.attribute_features(attrs).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        return out_im, out_obj, out_attrs


class OptimizedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(OptimizedBlock, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()
        self.downsample = downsample

    def forward(self, in_feat):
        x = in_feat
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x + self.shortcut(in_feat)

    def shortcut(self, x):
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return self.c_sc(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(ResBlock, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.activation = nn.ReLU()
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)

    def residual(self, in_feat):
        x = in_feat
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return x

    def forward(self, in_feat):
        return self.residual(in_feat) + self.shortcut(in_feat)


class CombineDiscriminator128(nn.Module):
    def __init__(self, num_classes=10, num_attrs=None):
        super(CombineDiscriminator128, self).__init__()
        self.obD = ResnetDiscriminator128(num_classes=num_classes, input_dim=3, num_attrs=num_attrs)

    def forward(self, images, bbox, label, attrs):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0), 1, 1).expand(-1, bbox.size(1), -1).float()
        bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        attrs = torch.cat((idx, attrs.float()), dim=2)
        bbox = bbox.view(-1, 5)
        attrs = attrs.view(-1, 81)

        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]

        attrs = attrs[idx]
        d_out_img, d_out_obj, d_out_attrs = self.obD(images, label, bbox, attrs)

        return d_out_img, d_out_obj, d_out_attrs