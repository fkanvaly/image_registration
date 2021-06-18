import torch
from scripts.mnist.utils import jacobian_det, multi_props_inj, Dice
import neurite as ne
from matplotlib import colors
import numpy as np


def evaluate_image(trainer, fix, moving, mode,  show=True):
    with torch.no_grad():
        if mode=="vxm":
            moved, flow = trainer.model(moving, fix)
        if mode=="inv":
            flow = trainer.model(moving, fix)
            moved = trainer.transform(moving, flow)

    jacobian = jacobian_det(flow)

    images = [img[0, 0, :, :].detach().numpy() for img in [moving, fix, moved]]
    titles = ['source', 'target', 'moved', 'jacobian', 'jacobian (binary)']
    cmaps = ['gray', 'gray', 'gray', 'Greens', 'Greens']
    norms = [colors.Normalize(vmin=vmin, vmax=vmax) for vmin, vmax in [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]]

    fig, axes = ne.plot.slices([*images, jacobian, 1 * (np.abs(jacobian) > 0.01)], titles=titles,
                               norms=norms, cmaps=cmaps, do_colorbars=True, show=show);
    ## Eval
    # print("Flow :")
    fig_flow, _ = ne.plot.flow([flow.squeeze().permute(1, 2, 0)], width=4, show=show);
    moved, fixed = moved[0, 0, :, :].detach().numpy(), fix[0, 0, :, :].detach().numpy()
    dice = Dice(moved, fixed)
    return {"fig": fig, 'flow':fig_flow, 'dice': dice}


def evaluate_inj(trainer, x_val, mode):
    inj_score = 100 * (1 - multi_props_inj(trainer, x_val, mode))
    return {"injectivity_indicator ", inj_score}