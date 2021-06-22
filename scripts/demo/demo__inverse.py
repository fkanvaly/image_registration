import streamlit as st
import sys

sys.path.append("./")


def app():
    st.write(r"""
    # Inverse-Consistent Deep Networks for Unsupervised Deformable Image Registration    Jun Zhang
    📝 Paper: [https://arxiv.org/pdf/1809.03443.pdf)
    
    ## **🥊 Proposed solution**
    
    This second model is based on the architecture of a  inverse-consistent deep neural network, as well as the objective 
    
    function (with both the proposed inverse-consistent and anti-folding constraints) for network training.
    
    Let's name A and B, the moving image and the fixed image, respectivel. Here, the idea is to deform image A to match the image B according to a dense flow (i.e., discrete displacement field) F_AB defined in the A space, while the image B is deformed to match the image A via another dense flow F_BA defined in the B space. Specifically, we propose an inverse-consistent regularization term to penalize the difference between two transformations from the respective inverse mappings.
    
    
    ## ⚙️ Architecture
    
    
    ![model](https://i.imgur.com/DGwHeOu.png)
    
    Here, the idea is to use once again the VoxelMorph architecture, and to adapt it to the constraitns mentionned in the paper.
    Therefore, we have to consider additionnal constraints that was not involved in our first model.
    
    The constraints are :
    - Inverse-consistent Constraint
    - Anti-folding Constraint
    - Smoothness Constraint

    
    ## **🦾 Results**
    
    We will analyse the performance of this model in the following sections, and note why it performs differently from the first.
""")


if __name__ == '__main__':
    app()
