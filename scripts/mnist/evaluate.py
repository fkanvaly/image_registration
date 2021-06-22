import torch
from scripts.mnist.utils import jacobian_det, multi_props_inj, Dice
import neurite as ne
from matplotlib import colors
import numpy as np
import time

def evaluate_image(trainer, fix, moving, mode,  show=True):
    with torch.no_grad():
        if mode=="vxm":
            moved, flow = trainer.model(moving, fix)
        if mode=="inv":
            flow = trainer.model(moving, fix)
            moved = trainer.transform(moving, flow)
    
    jacob_det = jacobian_det(flow)
    jacob_born = max(abs(jacob_det.min()), abs(jacob_det.max()))

    images = [img[0, 0, :, :].detach().numpy() for img in [moving, fix, moved]]
    titles = ['source', 'target', 'moved', 'jacobian_det', 'jacobian_det (binary)']
    cmaps = ['gray', 'gray', 'gray', 'bwr', 'Reds']
    norms = [colors.Normalize(vmin=vmin, vmax=vmax) for vmin, vmax in [[0, 1], [0, 1], [0, 1], [-jacob_born, jacob_born], [0,1]]]

    fig, axes = ne.plot.slices([*images, jacob_det, 1-1 * (np.abs(jacob_det) > 0.1)], titles=titles,
                               norms=norms, 
                               cmaps=cmaps, do_colorbars=True, show=show);
    ## Eval
    # print("Flow :")
    fig_flow, _ = ne.plot.flow([flow.squeeze().permute(1, 2, 0)], width=4, show=show);
    moved, fixed = moved[0, 0, :, :].detach().numpy(), fix[0, 0, :, :].detach().numpy()
    dice = Dice(moved, fixed)
    
    return {"fig": fig, 'flow':fig_flow, 'dice': dice}

