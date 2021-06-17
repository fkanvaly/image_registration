import streamlit as st

import sys

sys.path.append("./")


def app():
    st.write(r"""
    ### **🎯 What?**
    
    Pairwise medical image registration. Image registration, also known as image fusion or image matching, is the process of aligning two or more images based on image appearances. Medical Image Registration seeks to find an optimal spatial transformation that best aligns the underlying anatomical structures. in other words, is the process of finding is the process of finding optimal transformation that puts different images optimal transformation that puts different images into spatial correspondence.
    
    ### **❓ Why?**
    
    Medical Image Registration is used in many clinical applications such as image guidance, motion tracking, segmentation, dose accumulation, image reconstruction and so on.
    
    ### ☝🏻 **Pior work and SoTa**
    
    Modelled the matching as an optimisation problem with an objective function. 
    
    (-) cons : inference take time (few minutes by using GPU)
    
    """)



if __name__ == '__main__':
    app()
