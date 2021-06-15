import os
import torch
import numpy as np
import torch.nn as nn
from scripts.mnist.utils import to_nametuple, Grad2D

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm


class VoxelMNIST(nn.Module):
    def __init__(self, inshape, nb_features, ndim):
        super().__init__()
        self.unet = vxm.networks.Unet(inshape=inshape, nb_features=nb_features)
        self.flow = nn.Conv2d(nb_features[1][-1], ndim, 3, padding=1)
        self.transformer = vxm.layers.SpatialTransformer(inshape)

    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        x = self.unet(x)
        flow_field = self.flow(x)
        moving_transformed = self.transformer(source, flow_field)
        return moving_transformed, flow_field


def build_vxm(config):
    # une architecture
    nb_features = [
        [32, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]

    model = VoxelMNIST(config.inshape, nb_features, config.ndim)

    # prepare the model for training and send to device
    model.to(config.device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # prepare image loss
    if config.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif config.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % config.image_loss)

    L_sim = image_loss_func
    L_smooth = Grad2D('l2').loss

    return to_nametuple({'model': model, 'optimizer': optimizer,
                         'losses': {'sim': L_sim, 'smooth': L_smooth}})


def train_vxm(config, trainer, train_data):
    loss_hist = []
    # training loops
    zero_phi = np.zeros([config.batch_size_train, *config.inshape, config.ndim])

    for epoch in range(config.epochs):
        for step in range(config.steps_per_epoch):
            # generate inputs (and true outputs) and convert them to tensors
            x_fix, x_mvt = next(train_data['fix']), next(train_data['moving'])
            size = min(x_fix.shape[0], x_mvt.shape[0])
            x_fix, x_mvt = x_fix[:size], x_mvt[:size]  # because the remaining batch element can have diff size
            inputs = [x_mvt, x_fix]

            # run inputs through the model to produce a warped image and flow field
            x_pred, flow = trainer.model(*inputs)

            # calculate total loss
            loss_sim = trainer.losses['sim'](x_pred, x_fix)
            loss_smooth = config.λ * trainer.losses['smooth'](None, flow)
            loss = loss_sim + loss_smooth

            loss_info = 'loss: %.6f ' % loss.item()

            # backpropagate and optimize

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            # print step info

            if step % config.steps_per_epoch == 0:
                epoch_info = 'epoch: %04d' % (epoch + 1)
                step_info = ('step: %d/%d' % (step + 1, config.steps_per_epoch)).ljust(14)
                print('  '.join((epoch_info, step_info, loss_info)), flush=True)

        loss_hist.append(loss.item())

    return loss_hist
