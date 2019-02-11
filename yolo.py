import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

from utils import to_np
from utils import to_tensor
from utils import batch_iou

class Preprocess(nn.Module):
    """Pytorch module that normalizes data for a VGG network
    """
    # These values are taken from http://pytorch.org/docs/master/torchvision/models.html
    RGB_MEANS = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
    RGB_STDS = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

    def forward(self, x):
        """Normalize a single image or a batch of images

        Args:
            x: a pytorch Variable containing and float32 RGB image tensor with
              dimensions (batch_size x width x heigth x RGB_channels) or
              (width x heigth x RGB_channels).
        Returns:
            a torch Variable containing a normalized BGR image with shape
              (batch_size x BGR_channels x width x heigth)
        """
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        # x is batch * width * heigth *channels,
        # make it batch * channels * width * heigth
        if x.size(3) == 3:
            x = x.permute(0, 3, 1, 2)
        means = self.RGB_MEANS
        stds = self.RGB_STDS
        if x.is_cuda:
            means = means.cuda()
            stds = stds.cuda()
        x = (x - Variable(means)) / Variable(stds)
        return x

def select_indices(input, indices):
    # output[i, j, k, l] = input[i, j, k, indices[i, j, k], l]
    new_shape = (input.shape[0], input.shape[1], input.shape[2], input.shape[4])
    output = np.fromfunction(lambda i, j, k, l :
        input[np.array(i, dtype=int), np.array(j, dtype=int), np.array(k, dtype=int),
            indices[np.array(i, dtype=int), np.array(j, dtype=int), np.array(k, dtype=int)],
            np.array(l, dtype=int)], new_shape)
    return output

class Yolo(nn.Module):

    def __init__(self, model, out_features, grid=24, boxes=2, classes=0, intermediate=16384):
        super(Yolo, self).__init__()
        self.classes = classes
        self.model = model
        self.grid = grid
        self.boxes = boxes

        self.preprocess = Preprocess()

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(out_features, intermediate),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(intermediate, grid*grid*(5*boxes+classes))
#             nn.Sigmoid() -> seems reasonable, but gives worse results
        )

        self.coord = 5.0
        self.noobj = 0.5
#         self.I = torch.eye(self.classes)

#     def int_to_onehot(self, x):
#         shape = x.shape
#         shape += (-1,)
#         return (self.I[x.flatten()]).reshape(shape)

    def get_responsible(self, out):
        boxes = out.view(out.size(0), self.grid, self.grid, self.boxes, -1)
        conf = boxes[:, :, :, :, 4]

        resp = torch.argmax(conf, dim=3)
        return select_indices(boxes, resp)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x)
        x = x.view(x.size(0), self.grid, self.grid, -1)
        return x

    def loss(self, out, y):
#         boxes = out[:, :, :, :5*self.boxes]

        resp = self.get_responsible(out)
        y_conf = y[:, :, :, 4]

        E = resp - y
        E_coords = E[:, :, :, :4]
        E_conf   = E[:, :, :, 4]

        L_coords = self.coord * torch.sum(y_conf.unsqueeze(3) * E_coords * E_coords)
        L_conf = torch.sum((y_conf + (1-y_conf)*self.noobj) * E_conf * E_conf)

#         classes = out[:, :, :, 5*self.boxes:]
#         y_class = y[:, :, :, 5].squeeze()
#         y_onehot = self.int_to_onehot(y_class)

#         L += y_conf * torch.pow(classes-y_onehot, 2)

        return L_coords + L_conf

    def predict(self, out):
        return self.get_responsible(out)
