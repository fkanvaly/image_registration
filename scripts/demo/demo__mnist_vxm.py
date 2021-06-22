import streamlit as st
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./")

from scripts.mnist.data_loader import MNISTData
import os
from scripts.mnist.evaluate import evaluate_image
from scripts.mnist.voxelmorph import load_vxm

## Evaluate
import torch
import neurite as ne
from matplotlib import colors
import numpy as np

topology = {'0 hole': [1, 2, 3, 4, 5],
            '1 hole': [0, 9, 6],
            '2 holes': [8]
            }


@st.cache
def mnist():
    return MNISTData()


@st.cache
def st_load_test_data(src, dst):
    data = mnist().test_data(moving=src, fix=dst, dataset=True)
    return data


@st.cache(allow_output_mutation=True)
def st_load_model(name):
    path = os.path.join(f'output/model-mnist-vxm-{name}.pt')
    #conf, trainer, hist, inj = load_vxm("mnist", path)
    conf, trainer, hist = load_vxm("mnist", path)
    trainer.model.eval()
    return conf, trainer, hist #, inj


def digit_choice(k, same=False, src_d=None, dst_d=None):
    src_digit = [i for i in range(10)] if src_d is None else src_d
    dst_digit = src_digit if dst_d is None else dst_d

    col1, col2 = st.beta_columns(2)
    with col1:
        if len(dst_digit) > 1:
            src = st.selectbox(f'Source Digit | id:{k}', src_digit, 0)
        else:
            src = st.selectbox(f'Source Digit | id:{k}', src_digit)

    if same:
        dst_digit = [src]
    with col2:
        if len(dst_digit) > 1:
            dst = st.selectbox(f'Target Digit | id:{k + 1}', dst_digit, 1)
        else:
            dst = st.selectbox(f'Target Digit | id:{k + 1}', dst_digit)

    return src, dst


def double_slider(n1, n2, k):
    col1, col2 = st.beta_columns(2)
    with col1:
        idx1 = st.slider(f'Source data index | id:{k}', 0, n1, 1)
    with col2:
        idx2 = st.slider(f'Target data index | id:{k + 1}', 0, n2, 2)

    return idx1, idx2


def family(k):
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    return st.radio(f"Topology | id:{k}:", list(topology.keys()))


def double_family(k):
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    col1, col2 = st.beta_columns(2)
    with col1:
        src = st.radio(f"Source topology | id:{k}:", list(topology.keys()))
    with col2:
        dst = st.radio(f"Target topology | id:{k + 1}:", list(topology.keys()))
    return src, dst



def eval_model(model, data, k):
    N_fix, N_mvg = len(data['fix']), len(data['moving'])
    idx1, idx2 = double_slider(N_fix, N_mvg, k)
    val_fix = data['fix'][idx1].unsqueeze(0)
    val_mvt = data['moving'][idx2].unsqueeze(0)
    res = evaluate_image(model, val_mvt, val_fix, mode="vxm", show=False)
    col1, col2 = st.beta_columns([5,2])
    with col1:
        st.pyplot(res['fig'])
    with col2:
        st.pyplot(res['flow'])
    st.write("Dice score :", res['dice'])

    


def app():
    st.write(r"""
    # VoxelMorph - Baseline test on MNIST
    ## Result summary
    """)
    
    "We have trained the voxelmorph model with different results and here are the conclusions. "
    
    r"""
    For this model the most relevant parameter was $\lambda$. It represents the regularisation that is applied to force the model to generate diffeomorphic displacement fields.
    """
    
    r"### 1. $\lambda = 0$"
    col1, col2 = st.beta_columns([5,2])
    with col1:
        st.image('https://i.imgur.com/pcJ6jzD.png', 'result of image transformation')
    with col2:
        st.image('https://i.imgur.com/Y3zDgJh.png', 'flow')
    
    r"""
    Here we see that without any constraint on smoothness we have a good dice score, but the flow is complicated and the jacobian determinant has several pixels (the brown ones) which are close to zero. It shows that it is not very diffeomorphic and we want it to be diffeomorphic because it is important on brain images.
    
    We finaly we make two evaluation with our two metrics (DICE & Injectivity score) and we got:
    - Dice score : 0.9262536873156342
    - Injectivity indicator : $\approx 50\%$
    """
    
    r"### 1. $\lambda = 0.5$"
    col1, col2 = st.beta_columns([5,2])
    with col1:
        st.image('https://i.imgur.com/GokdrmY.png', 'result of image transformation')
    with col2:
        st.image('https://i.imgur.com/CBeralz.png', 'flow')
    
    r"""
    We see that with regularization on the smoothness the flow is more controlled and we don't have any pixel which is not "diffeomorphic". The injectivity inducator is much higher. Therefore the registration is more complicated and the moved image doesn't look like the fixed one but on brain images the images won't so different. 
    
    We finaly we make two evaluation with our two metrics (DICE & Injectivity score) and we got:
    - Dice score : 0.789873417721519
    - Injectivity indicator : $\approx 85\%$
    """
    

    
    "##  Experiment 🧪"
    exp = st.checkbox('Do experiement?', True)
    if exp:
        st.write(r"""
        ### 🧠 Load model
         """)

        pattern = "model-mnist-vxm"
        model_availbale = [ filename[len(pattern)+1:-3] for filename in os.listdir('./output') if pattern in filename]
        name = st.selectbox('config:', model_availbale)

        #### Load model
        #conf, model, hist, inj = st_load_model(name)
        conf, trainer, hist = st_load_model(name)
        st.write(conf)
        #st.write("Injectivity indicator:", inj)
        fig = plt.figure(figsize=(7,5))
        plt.plot(np.arange(len(hist)), hist)
        plt.title("loss")
        st.pyplot(fig)

        st.write("""### 🧪 Evaluation 1 - Validation set""")
        agree1 = st.checkbox('Display ? id:1', True)
        if agree1:
            data1 = st_load_test_data(conf.fix, conf.moving)
            eval_model(trainer, data1, 0)

        st.write("""### 🧪 Evaluation 2 - Generalization""")
        st.write("We can classify the numbers into 3 family: `0 hole`, `1 hole`, `2 holes`")
        for k, v in topology.items():
            st.write(f"`{k}`: {v}")

        st.write("""
        ___
        ### 〰️ Same topology transfert
        """)
        st.write("""- _** same number type**_ 1️⃣ -> 1️⃣""")
        agree2 = st.checkbox('Display ? id:2', True)
        if agree2:
            fam_name = family(0)
            topo1 = topology[fam_name]
            src, _ = digit_choice(src_d=topo1, dst_d=topo1, k=0, same=True)
            data2 = st_load_test_data(src, src)
            eval_model(trainer, data2, 1)

        st.write("""- _** different number type**_ 1️⃣ ->2️⃣""")
        agree3 = st.checkbox('Display ? id:3', True)
        if agree3:
            fam_name = family(1)
            topo1 = topology[fam_name]
            src, dst = digit_choice(src_d=topo1, dst_d=topo1, k=1)
            data3 = st_load_test_data(dst, src)
            eval_model(trainer, data3, 2)

        st.write("""
        ___
        ### ♾️ Different topology transfert
        """)
        agree4 = st.checkbox('Display ? id:4', True)
        if agree4:
            src_fam_name, dst_fam_name = double_family(2)
            topo1 = topology[src_fam_name]
            topo2 = topology[dst_fam_name]

            src, dst = digit_choice(src_d=topo1, dst_d=topo2, k=2)
            data4 = st_load_test_data(dst, src)
            eval_model(trainer, data4, 3)


if __name__ == '__main__':
    app()
