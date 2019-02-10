import torch
import torchvision
import numpy as np

import os
import gdown
import subprocess

from utils import DogsDetectionDataset
from utils import SubsampledImageDataset
from utils import to_np
from utils import to_tensor
from utils import batch_iou

from yolo import Yolo

CUDA = True

# fetch the data
if not os.path.isfile('./data.tar.gz'):
    gdown.download('https://drive.google.com/uc?export=download&id=1WSWZyDZkmwpUkgCq0rKeDRag7pT7yM1w', 'data.tar.gz', False)
if not os.path.isfile('./yolo_metadata.npy'):
    gdown.download('https://drive.google.com/uc?export=download&id=1hOOxP6jVpIB3bO4i9hZNS13kTXPcMuDb', 'yolo_metadata.npy', False)
if not os.path.isdir('./data'):
    subprocess.call("tar zxf data.tar.gz", shell=True)

dogs_dataset = DogsDetectionDataset('./yolo_metadata.npy', './raw', CUDA=CUDA)

all_indexes = np.random.permutation(len(dogs_dataset))
train_size = int(len(all_indexes) * 0.8)

dogs_train_dataset = SubsampledImageDataset(
    dogs_dataset,
    all_indexes[:train_size])

dogs_test_dataset = SubsampledImageDataset(
    dogs_dataset,
    all_indexes[train_size:])

dogs_datasets = {
    'train': dogs_train_dataset,
    'test': dogs_test_dataset,
}

batch_size = 64

dogs_loaders = {
    'train': torch.utils.data.DataLoader(
        dogs_datasets['train'], batch_size=batch_size,
        num_workers=0,
        shuffle=True),
    'test': torch.utils.data.DataLoader(
        dogs_datasets['test'], batch_size=batch_size,
        num_workers=0,
        shuffle=False)}

# Training code is based on assignment 4
def compute_error_rate(model, data_loader, cuda=True):
    model.eval()
    iou_sum = 0
    num_examples = 0
    for x, y in data_loader:
        if cuda:
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            out = model.forward(x)
            pred = model.predict(out)
            iou_sum += torch.sum(batch_iou(y, pred))
            num_examples += x.size(0)
    return 100.0 * (1.0 - iou_sum / num_examples)


def train(model, data_loaders, optimizer, num_epochs=1,
          log_every=100, cuda=True):
    if cuda:
        model.cuda()
    iter_ = 0
    epoch = 0
    best_params = None
    best_val_err = np.inf
    history = {'train_losses': [], 'train_errs': [], 'val_errs': [], 'train_loss2': []}
    print('Training the model!')
    print('You can interrupt it at any time.')
    while epoch < num_epochs:
        model.train()
        # model.train_mode()
        epoch += 1
        for x, y in data_loaders['train']:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            iter_ += 1

            optimizer.zero_grad()
            out = model.forward(x)
            loss = model.loss(out, y)
            loss.backward()
            optimizer.step()

            pred = model.predict(out)
            iou = torch.sum(batch_iou(y, pred))
            err_rate = 100 * (1.0 - iou / out.size(0))

            history['train_losses'].append(loss.item())
            history['train_errs'].append(err_rate.item())

            if iter_ % log_every == 0:
                print "Minibatch {0: >6}  | loss {1: >5.2f} | err rate {2: >5.2f}%" \
                      .format(iter_, loss.item(), err_rate.item())

        val_err_rate = compute_error_rate(model, data_loaders['test'], cuda)
        history['val_errs'].append((iter_, val_err_rate.item()))

        if val_err_rate < best_val_err:
            best_epoch = epoch
            best_val_err = val_err_rate
        m = "After epoch {0: >2} | valid err rate: {1: >5.2f}% | doing {2: >3} epochs" \
            .format(epoch, val_err_rate, num_epochs)
        print '{0}\n{1}\n{0}'.format('-' * len(m), m)

resnet = torchvision.models.resnet50(pretrained=True)
out_features = resnet.fc.in_features
yolo = Yolo(resnet, out_features)

# Train only the params of the classifier
optimizer = torch.optim.Adam(yolo.model.fc.parameters(), lr=1e-5, weight_decay=1e-3)

train(yolo, dogs_loaders, optimizer,
      num_epochs=150, log_every=50, cuda=CUDA)
