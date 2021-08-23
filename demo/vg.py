from collections import OrderedDict
import os

from model.resnet_generator import ResnetGenerator128 as ResnetGenerator128v1
from model.resnet_generator_v2 import ResnetGenerator128 as ResnetGenerator128v2
from model.resnet_generator_v2 import ResnetGenerator256 as ResnetGenerator256v2

from utils.util_v2 import *


def sample(version, img_size,
           bbox, label, attr,
           z_im=None, z_obj=None, resample=True):
    num_classes = 179
    num_o = 31
    num_attrs = 80

    if version == 1:
        assert img_size == 128, "args.img_size should be 128 for version 1"

        netG = ResnetGenerator128v1(num_classes=num_classes, output_dim=3, num_attrs=num_attrs)
        model_path = './pretrained/AttrLostGANv1_128x128.pth'
    elif version == 2:
        if img_size == 128:
            netG = ResnetGenerator128v2(num_classes=num_classes, output_dim=3, num_attrs=num_attrs)
            model_path = './pretrained/AttrLostGANv2_128x128.pth'
        elif img_size == 256:
            netG = ResnetGenerator256v2(num_classes=num_classes, output_dim=3, num_attrs=num_attrs)
            model_path = './pretrained/AttrLostGANv2_256x256.pth'
        else:
            assert False, "args.img_size should be 128 or 256"
    else:
        assert False, "args.version should be 1 or 2"

    if not os.path.isfile(model_path):
        print('model not found')
        print(model_path)
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
    bbox = np.asarray(bbox) / img_size
    arr = np.zeros(num_attrs, dtype=float)
    arr.fill(-1)

    bbox = np.vstack((bbox, np.array([0, 0, 1, 1])))
    label = np.hstack((label, [0]))
    attr = np.vstack((attr, [np.zeros(num_attrs, dtype=float)]))

    for i in range(len(bbox), num_o):
        label = np.hstack((label, [0]))
        attr = np.vstack((attr, [arr]))
        bbox = np.vstack((bbox, np.array([-0.6, -0.6, 0.5, 0.5])))

    bbox = torch.from_numpy(np.asarray(bbox)).unsqueeze(0)
    label = torch.from_numpy(np.asarray(label))
    attr = torch.from_numpy(np.asarray(attr)).unsqueeze(0).float()  

    label = label.long()

    if resample:
        z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float()
        z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float()

    fake_images = netG.forward(z_obj, bbox, z_im,
                               label.unsqueeze(dim=0),
                               attr)

    image = fake_images[0]
    image = image.detach().numpy()
    image = image.transpose(1, 2, 0)
    #image = image 0.5 + 0.5
    #image = image * 255
    return image
