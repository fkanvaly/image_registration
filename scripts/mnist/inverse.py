import os
import torch
import numpy as np
import torch.nn as nn
from scripts.mnist.utils import to_nametuple, mse_loss, smoothloss, antifoldloss
from scripts.mnist.data_loader import MNISTData
from torchsummary import summary
from tqdm import tqdm

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm


class UnetMNIST(nn.Module):
    def __init__(self, inshape, nb_features, ndim):
        super().__init__()
        self.unet = vxm.networks.Unet(inshape=inshape, nb_features=nb_features)
        self.flow = nn.Conv2d(nb_features[1][-1], ndim, 3, padding=1)

    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        x = self.unet(x)
        flow_field = self.flow(x)
        return flow_field


class SpatialT(nn.Module):
    def __init__(self, inshape):
        super().__init__()
        self.transformer = vxm.layers.SpatialTransformer(inshape)

    def forward(self, source, flow_field):
        moving_transformed = self.transformer(source, flow_field)
        return moving_transformed


def build_inverse(config, device="cpu"):
    # une architecture
    nb_features = [
        [32, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]

    model = UnetMNIST(config.inshape, nb_features, config.ndim)
    transform = SpatialT(config.inshape)
    # prepare the model for training and send to device
    model.to(device)
    transform.to(device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    return to_nametuple({'model': model, 'transform': transform, 'optimizer': optimizer,
                         'losses': {'sim': mse_loss,
                                    'inv': mse_loss,
                                    'antifold': antifoldloss,
                                    'smooth': smoothloss}})


def train_inverse(config, trainer, train_data, verbose=True, device="cpu"):
    lossall = np.zeros((5, config.epochs))

    for epoch in tqdm(range(config.epochs)):
        for step in range(config.steps_per_epoch):
            # generate inputs (and true outputs) and convert them to tensors
            x_fix, x_mvt = next(train_data['fix']), next(train_data['moving'])
            size = min(x_fix.shape[0], x_mvt.shape[0])
            X, Y = x_fix[:size].to(device), x_mvt[:size].to(device)  # because the remaining batch element can have diff size

            F_xy = trainer.model(*[X, Y])
            F_yx = trainer.model(*[Y, X])

            X_Y = trainer.transform(X, F_xy)
            Y_X = trainer.transform(Y, F_yx)

            F_xy_ = trainer.transform(-F_yx, F_yx)
            F_yx_ = trainer.transform(-F_xy, F_xy)

            loss1 = trainer.losses['sim'](Y, X_Y) + trainer.losses['sim'](X, Y_X)  # range_flow ?
            loss2 = config.inverse * (trainer.losses['inv'](F_xy, F_xy_) + trainer.losses['inv'](F_yx, F_yx_))
            loss3 = config.antifold * (trainer.losses['antifold'](F_xy) + trainer.losses['antifold'](F_yx))
            loss4 = config.smooth * (trainer.losses['smooth'](F_xy) + trainer.losses['smooth'](F_yx))

            loss = loss1 + loss2 + loss3 + loss4
            loss_info = 'loss: %.6f ' % loss.item()

            # back propagate and optimize
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            # print step info
            if verbose:
                if step % config.steps_per_epoch == 0:
                    epoch_info = 'epoch: %04d' % (epoch + 1)
                    step_info = ('step: %d/%d' % (step + 1, config.steps_per_epoch)).ljust(14)
                    print('  '.join((epoch_info, step_info, loss_info)), flush=True)

        lossall[:,epoch] = np.array([loss.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item()])

    return lossall


def load_inverse(path, device='cpu'):
    checkpoint = torch.load(path)
    conf = to_nametuple(checkpoint['config'])

    trainer = build_inverse(conf)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    trainer.model.to(device)
    print(f'model load on device: {device}')

    return conf, trainer


def train(conf, device="cpu", save=True, save_name='default', save_folder='output', verbose=True):
    print(f'train on {device}')
    if device=="cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.backends.cudnn.deterministic = True
        
    # load data
    mnist_data = MNISTData()
    x_train, x_val = mnist_data.train_val(conf.fix, conf.moving)

    # build model
    trainer = build_inverse(conf, device)

    if verbose:
        print(summary(trainer.model, [(1, *conf.inshape), (1, *conf.inshape)]))

    # train model
    train_inverse(conf, trainer, x_train, verbose, device)

    # save model
    if save:
        torch.save({'config': dict(conf._asdict()),
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    }, os.path.join(save_folder, f'model-inverse-{save_name}.pt'))

    return trainer
