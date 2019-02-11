import torch
import numpy as np

import os
import PIL
from PIL import Image

import torchvision
from torch.utils.data import Dataset, DataLoader

def to_np(x):
    return x.detach().cpu().numpy()

def to_tensor(x, CUDA, **kwargs):
    x = torch.from_numpy(x)
    if CUDA:
        x = x.cuda()
    return x.clone()

class DogsDetectionDataset(Dataset):

    def __init__(self, meta_file, root_dir, CUDA=True, imsize=224):
        self.meta = np.load(meta_file).astype(np.float32)
        self.root = root_dir
        self.imsize = imsize
        self.CUDA = CUDA

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, str(idx) + '.jpg')
        image = Image.open(img_name)
        image = image.resize((self.imsize, self.imsize), PIL.Image.ANTIALIAS)

        torch_img = to_tensor(np.array(image, dtype=np.float32), self.CUDA)
        torch_meta = to_tensor(self.meta[idx], self.CUDA)
        return (torch_img, torch_meta)

class SubsampledImageDataset(Dataset):
    def __init__(self, dataset, indexes, **kwargs):
        super(SubsampledImageDataset, self).__init__(**kwargs)
        self.dataset = dataset
        self.indexes = indexes

    def __getitem__(self, i):
        img, label = self.dataset[self.indexes[i]]
        return img, label

    def __len__(self):
        return len(self.indexes)

def batch_label_to_bbox(label):
    grid = label.shape[1]
    stride = 1.0 / float(grid)

    label = label.view(label.shape[0], -1, 5)
    conf = label[:, :, 4]

    best = torch.argmax(conf, dim=1)
    pred = []
    for i in range(label.shape[0]):
        b = best[i]
        x, y = int(b % grid), int(b / grid)
        coords = label[i, b, :]

        box_x, box_y = stride * (x+coords[0]), stride * (y+coords[1])
        box_w, box_h = coords[2]*coords[2], coords[3]*coords[3]

        pred.append(torch.FloatTensor([box_x, box_y, box_w, box_h]))

    return torch.stack(pred, dim=0)

def batch_change_repr(batch):
    mid_xy = batch[:, :2]
    wh = batch[:, 2:4]
    topleft_xy = mid_xy - wh/2
    bottomright_xy = mid_xy + wh/2
    return torch.cat((topleft_xy, bottomright_xy), 1)

def batch_iou(pred, truth):
    pred = batch_label_to_bbox(pred)
    pred_xy = pred[:, :2]
    pred_wh = pred[:, 2:4]

    truth = batch_label_to_bbox(truth)
    truth_xy = truth[:, :2]
    truth_wh = truth[:, 2:4]


#     stride = 1.0 / 24.0
#     x_pred, y_pred = torch.floor(torch.div(pred[:, 0], stride)), torch.floor(torch.div(pred[:, 1], stride))
#     x_truth, y_truth = torch.floor(torch.div(truth[:, 0], stride)), torch.floor(torch.div(truth[:, 1], stride))

#     x_score = torch.eq(x_pred, x_truth)
#     y_score = torch.eq(y_pred, y_truth)

#     return (x_score * y_score).type(pred.type())

    pred_area = torch.prod(pred_wh, dim=1)
    truth_area = torch.prod(truth_wh, dim=1)

    pred = torch.cat((pred_xy, pred_wh), dim=1)
    pred = batch_change_repr(pred)
    truth = torch.cat((truth_xy, truth_wh), dim=1)
    truth = batch_change_repr(truth)

    both = torch.stack((pred, truth), dim=2)

    topleft = torch.max(both[:, :2, :], dim=2)[0]
    bottomright = torch.min(both[:, 2:, :], dim=2)[0]

    inter_wh = torch.clamp(bottomright - topleft, min=0.0)
    inter_area = torch.prod(inter_wh, dim=1)

    return inter_area / (pred_area + truth_area - inter_area)
