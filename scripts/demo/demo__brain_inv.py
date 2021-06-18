import streamlit as st
import sys
import numpy as np

sys.path.append("./")

from scripts.mnist.data_loader import BrainData
from config import inverse
import os
from scripts.mnist.evaluate import evaluate_image
from scripts.mnist.inverse import load_inverse

topology = {'0 hole': [1, 2, 3, 4, 5],
            '1 hole': [0, 9, 6],
            '2 holes': [8]
            }


@st.cache
def st_load_data():
    return BrainData().test_data(dataset=True)


@st.cache(allow_output_mutation=True)
def st_load_model(name):
    path = os.path.join(f'output/model-brain-inverse-{name}.pt')
    conf, trainer, hist = load_vxm(path)
    trainer.model.eval()
    return conf, trainer, hist


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
    res = evaluate_image(model, val_fix, val_mvt, "inv", show=False)
    st.pyplot(res['fig'])


def app():
    st.write(r"""
    # Inverse-Consistent - Brain Image
    ##  Model for same source distribution 1️⃣ $\rightarrow$ 1️⃣
    ### 🧠 Load model
     """)
    
    name = st.selectbox('config: λ = 0.5 -> lambda-0_5', list(inverse.keys()))

    #### Load model
    conf, trainer, hist = st_load_model(name)
    st.write(conf)

    st.write("""### 🧪 Evaluation - Validation set""")
    agree1 = st.checkbox('Display ? id:1')
    if agree1:
        data1 = st_load_test_data(conf.fix, conf.moving)
        eval_model(trainer, data1, 0)



if __name__ == '__main__':
    app()
