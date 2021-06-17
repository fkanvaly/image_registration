import streamlit as st
import sys


sys.path.append("./")


def app():
    st.write(r"""
    # VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    📝 Paper: [https://arxiv.org/pdf/1809.05231.pdf](https://arxiv.org/pdf/1809.05231.pdf)
    
    ## **🥊 Proposed solution**
    
    Use CNN to learn to generate the deformable field. Then applied it the the moving images.
    
    (+) pros : Fast inference (under second using GPU), accuracy close to sota
    
    ## ⚙️ Architecture
    
    The deformation field is generate by U-net architecture
    
    ![model](https://i.imgur.com/IFkkNRZ.png)
    
    The loss can be decomposed in two component : A image level similarly measure and the deformation field smoothness measure.
    
    There two possible similarity measure : MSE and Cross-correlation
    
    
    ## **🦾 Results**
    
    **Metrics :** The evaluation metric is the Dice score on anatomical segmentation. This measure in a certain way measure the volume overlap.
    
    **Experimental setup :** They did an atlas based registration. This means that we take the atlas model of the brain as reference and we try to transform our input to that reference.
    
    They use an atlas computed using an external dataset . Each input volume pair consists of the atlas (image f) and a volume from the dataset (image m). image pairs using the same fixed atlas for all examples. 
    
     ![model](https://i.imgur.com/JGyaLyu.png)
     """)
    


if __name__ == '__main__':
    app()
