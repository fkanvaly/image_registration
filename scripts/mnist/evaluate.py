import torch
from scripts.mnist.utils import jacobian_det, multi_props_bij, Dice
import neurite as ne
from matplotlib import colors
import numpy as np


def evaluate_vxm(model, fix, moving, show=True):
    with torch.no_grad():
        moved, flow = model(moving, fix)

    jacobian = jacobian_det(flow)

    images = [img[0, 0, :, :].detach().numpy() for img in [moving, fix, moved]]
    titles = ['source', 'target', 'moved', 'jacobian', 'jacobian (binary)']
    cmaps = ['gray', 'gray', 'gray', 'Greens', 'Greens']
    norms = [colors.Normalize(vmin=vmin, vmax=vmax) for vmin, vmax in [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]]

    fig, axes = ne.plot.slices([*images, jacobian, 1 * (np.abs(jacobian) > 0.01)], titles=titles,
                               norms=norms, cmaps=cmaps, do_colorbars=True);
    ## Eval
    # print("Flow :")
    ne.plot.flow([flow.squeeze().permute(1, 2, 0)], width=4, show=show);
    # inj_score = 100 * (1 - multi_props_bij(trainer.model, data))
    # print("Injectivity indicator : ", inj_score)
    moved, fixed = moved[0, 0, :, :].detach().numpy(), fix[0, 0, :, :].detach().numpy()
    dice = Dice(moved, fixed)
    # print("DICE score %.2f" % dice)

    return {"fig": fig, 'score': {'dice': dice,
                                  # 'inj': inj_score
                                  }}


def evaluate_inv(trainer, fix, moving, show=True):
    with torch.no_grad():
        flow = trainer.model(moving, fix)
        moved = trainer.transform(moving, flow)

    jacobian = jacobian_det(flow)

    images = [img[0, 0, :, :].detach().numpy() for img in [moving, fix, moved]]
    titles = ['source', 'target', 'moved', 'jacobian', 'jacobian (binary)']
    cmaps = ['gray', 'gray', 'gray', 'RdBu', 'RdYlGn']
    norms = [colors.Normalize(vmin=vmin, vmax=vmax) for vmin, vmax in [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]]

    fig, axes = ne.plot.slices([*images, np.abs(jacobian), 1 * (np.abs(jacobian) < 0.01)], titles=titles,
                               norms=norms, cmaps=cmaps, do_colorbars=True);

    ## Eval
    # print("Flow :")
    ne.plot.flow([flow.squeeze().permute(1, 2, 0)], width=4, show=show);
    # inj_score = 100 * (1 - multi_props_bij(trainer.model, data))
    # print("Injectivity indicator : ", inj_score)
    moved, fixed = moved[0, 0, :, :].detach().numpy(), fix[0, 0, :, :].detach().numpy()
    dice = Dice(moved, fixed)
    # print("DICE score %.2f" % dice)

    return {"fig": fig, 'score': {'dice': dice,
                                  # 'inj': inj_score
                                  }}
