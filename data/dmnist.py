import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from ast import literal_eval


class DialogMNISTDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=0)
        self.root_dir = root_dir
        self.transform = transform

        self.annotations['labels'] = self.annotations['labels'].apply(literal_eval)
        self.annotations['bbox'] = self.annotations['bbox'].apply(literal_eval)
        self.annotations['color'] = self.annotations['color'].apply(literal_eval)
        self.annotations['bgcolor'] = self.annotations['bgcolor'].apply(literal_eval)
        self.annotations['style'] = self.annotations['style'].apply(literal_eval)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.annotations.loc[idx, 'name'])
        image = Image.open(img_name).convert('RGB')
        labels = self.annotations.loc[idx, 'labels']
        bbox = self.annotations.loc[idx, 'bbox']
        bbox = np.vstack(bbox)
        color = self.annotations.loc[idx, 'color']
        bgcolor = self.annotations.loc[idx, 'bgcolor']
        style = self.annotations.loc[idx, 'style']

        """ **Encode attributes** 
        Build indices of list of possible attributes.
        Encode given attributes in the batch based on these indices. E.g ['blue', 'red', 'green'] -> [1 2 3]
        Can use nn.Embeddings later to embed these encoded attributes.
        """
        color_list = ['blue', 'red', 'green', 'violet', 'brown']
        color_list_to_idx = {col: i for i, col in enumerate(color_list)}
        enc_color = [color_list_to_idx[c] for c in color]

        bgcolor_list = ['white', 'cyan', 'salmon', 'yellow', 'silver']

        bgcolor_list_to_idx = {bgcol: i for i, bgcol in enumerate(bgcolor_list)}
        enc_bgcolor = [bgcolor_list_to_idx[bgc] for bgc in bgcolor]

        style_list = ['flat', 'stroke']
        style_list_to_idx = {stl: i for i, stl in enumerate(style_list)}
        enc_style = [style_list_to_idx[s] for s in style]
        bbox = np.true_divide(bbox, np.array([128, 128, 128, 128]))

        for _ in range(len(bbox), 8):  # append 0 to ensure shape is max 8 = num digits on the canvas
            labels = np.hstack((labels, [10]))
            enc_bgcolor = np.hstack((enc_bgcolor, [5]))
            enc_color = np.hstack((enc_color, [5]))
            enc_style = np.hstack((enc_style, [2]))
            bbox = np.vstack((bbox, np.array([-0.6, -0.6, 0.5, 0.5])))

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.LongTensor(labels)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        enc_color = torch.LongTensor(enc_color)
        enc_bgcolor = torch.LongTensor(enc_bgcolor)
        enc_style = torch.LongTensor(enc_style)
        return image, labels, bbox, enc_color, enc_bgcolor, enc_style

