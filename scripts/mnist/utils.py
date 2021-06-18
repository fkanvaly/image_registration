import torch
import numpy as np
from collections import namedtuple

import torch.nn.functional as F


######################################
#           DATA UTILS               #
######################################

def to_nametuple(d):
    Param = namedtuple('Param', d)
    return Param(**d)


class Grad2D:
    """
    Compute the gradient for 2D data
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad


def prop_inj(flow):
    size = flow.shape[2:]
    n = flow.shape[2]
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor)

    new_locs = grid + flow
    new_locs = new_locs.squeeze().round().clip(0, n)

    # ou %32
    true_total = new_locs.shape[1] * new_locs.shape[2]
    current_unique = len(set(zip(new_locs[0].flatten().tolist(), new_locs[1].flatten().tolist())))
    nb_same = true_total - current_unique

    return nb_same / true_total


def multi_props_inj(trainer, x_val, mode):
    props = np.zeros(20)
    for k in range(len(props)):
        val_fix = next(x_val['moving'])[:1, ...]
        val_mvt = next(x_val['moving'])[:1, ...]
    

        with torch.no_grad():
            if mode=="vxm":
                x_val_pred, flow_val_pred = trainer.model(val_mvt, val_fix)
            if mode=="inv":
                flow_val_pred = trainer.model(val_mvt, val_fix) 
                x_val_pred = trainer.transform(val_mvt, flow_val_pred)
            props[k] = prop_bij(flow_val_pred)
    return props.sum() / len(props)


def Dice(segm, segm_ref):
    if segm.shape != segm_ref.shape:
        raise ValueError("Les deux images doivent avoir la meme taille.")
    segm = np.asarray(segm).astype(np.bool)
    segm_ref = np.asarray(segm_ref).astype(np.bool)
    intersection = np.logical_and(segm, segm_ref)
    dice = 2.0 * intersection.sum() / (segm.sum() + segm_ref.sum())
    return dice

#new:
#inference time:  0.04457521438598633 s
#jacob time:  0.0006272792816162109 s
#dice time:  0.00046324729919433594 s
def jacobian_det(flow):
    size = flow.shape[2:]
    n = flow.shape[2]
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor)

    new_locs = (grid + flow).squeeze()
    jac_det = np.zeros(size)
    
    n1, n2 = size[0]-1, size[1]-1
    dx = new_locs[:, :, 1:] - new_locs[:, :, :-1]
    dy = new_locs[:, 1:, :] - new_locs[:, :-1, :]
    A, C = dx[0,:][:n1,:n2], dx[1,:][:n1,:n2]
    B, D = dy[0,:][:n1,:n2], dy[1,:][:n1,:n2]
    det = A*D - B*C
    
    return det.numpy()

#old : 
#inference time:  0.04196906089782715 s
#jacob time:  4.360510349273682 s
#dice time:  0.00041031837463378906 s
def jacobian_det_old(flow):
    size = flow.shape[2:]
    n = flow.shape[2]
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor)

    new_locs = grid + flow
    jac_det = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            if i == 0:
                A = new_locs.squeeze()[0][i + 1, j] - new_locs.squeeze()[0][i, j]
                C = new_locs.squeeze()[1][i + 1, j] - new_locs.squeeze()[1][i, j]
            if i == size[0] - 1:
                A = new_locs.squeeze()[0][i, j] - new_locs.squeeze()[0][i - 1, j]
                C = new_locs.squeeze()[1][i, j] - new_locs.squeeze()[1][i - 1, j]
            if j == 0:
                B = new_locs.squeeze()[0][i, j + 1] - new_locs.squeeze()[0][i, j]
                D = new_locs.squeeze()[1][i, j + 1] - new_locs.squeeze()[1][i, j]
            if j == size[1] - 1:
                B = new_locs.squeeze()[0][i, j] - new_locs.squeeze()[0][i, j - 1]
                D = new_locs.squeeze()[1][i, j] - new_locs.squeeze()[1][i, j - 1]
            if 0 < i < size[0] - 1 and 0 < j < size[1] - 1:
                A = (new_locs.squeeze()[0][i + 1, j] - new_locs.squeeze()[0][i - 1, j]) / 2
                B = (new_locs.squeeze()[0][i, j + 1] - new_locs.squeeze()[0][i, j - 1]) / 2
                C = (new_locs.squeeze()[1][i + 1, j] - new_locs.squeeze()[1][i - 1, j]) / 2
                D = (new_locs.squeeze()[1][i, j + 1] - new_locs.squeeze()[1][i, j - 1]) / 2

            jac_det[i, j] = A * D - B * C
    return jac_det




def smoothloss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    return (torch.mean(dx * dx) + torch.mean(dy * dy)) / 2.0


def antifoldloss(y_pred):
    dy = y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :] - 1
    dx = y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:] - 1

    dy = F.relu(dy) * torch.abs(dy * dy)
    dx = F.relu(dx) * torch.abs(dx * dx)
    return (torch.mean(dx) + torch.mean(dy)) / 2.0


def mse_loss(input, target):
    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    diff = y_true_f - y_pred_f
    mse = torch.mul(diff, diff).mean()
    return mse
