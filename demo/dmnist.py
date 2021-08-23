from collections import OrderedDict
import os

from model.resnet_generator_dmnist import *
from utils.util_v2 import *


def sample(bbox, label, dcolor, bgcolor, dstyle,
           model_path, z_im=None, z_obj=None, resample=True):
    num_classes = 11
    num_o = 8
    num_dcolor = 6
    num_bgcolor = 6
    num_dstyle = 3

    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3,
                              num_dcolor=num_dcolor, num_bgcolor=num_bgcolor, num_dstyle=num_dstyle).cpu()

    if not os.path.isfile(model_path):
        print('model not found')
        return
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.eval()

    thres = 2.0
    bbox = np.asarray(bbox) / 128
    for _ in range(len(bbox), 8):
        label = np.hstack((label, [10]))
        dcolor = np.hstack((dcolor, [5]))
        bgcolor = np.hstack((bgcolor, [5]))
        dstyle = np.hstack((dstyle, [2]))
        bbox = np.vstack((bbox, np.array([-0.6, -0.6, 0.5, 0.5])))

    bbox = torch.from_numpy(np.asarray(bbox)).unsqueeze(0)
    label = torch.from_numpy(np.asarray(label)).unsqueeze(0)
    dcolor = torch.from_numpy(np.asarray(dcolor)).unsqueeze(0).long()
    bgcolor = torch.from_numpy(np.asarray(bgcolor)).unsqueeze(0).long()
    dstyle = torch.from_numpy(np.asarray(dstyle)).unsqueeze(0).long()

    label = label.unsqueeze(-1).long()

    if resample:
        z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float()
        z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float()

    fake_images = netG.forward(z_obj, bbox, z_im,
                               label.squeeze(dim=-1),
                               dcolor, bgcolor, dstyle)

    image = fake_images[0]
    image = image.detach().numpy()
    image = image.transpose(1, 2, 0)
    #image = image * 0.5 + 0.5
    #image = image * 255
    return image
