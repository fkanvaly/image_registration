
# Projet 5 - IMA206 - Image Registration 🧠

- 🎯 What ? : Pairwise medical image registration. Image registration, also known as image fusion or image matching, is the process of aligning two or more images based on image appearances. Medical Image Registration seeks to find an optimal spatial transformation that best aligns the underlying anatomical structures. in other words, is the process of finding is the process of finding optimal transformation that puts different images optimal transformation that puts different images into spatial correspondence.
- 🥊 Proposed solution : 
  - https://arxiv.org/pdf/1809.05231.pdf
  - https://arxiv.org/pdf/1809.03443.pdf


## Authors

- [Kanvaly Fadiga](https://kanvaly.io)
- [Benoit Marchandot](https://github.com/benoitmarchandot)
- [Mona Mokart](https://github.com/monamokart)
- [Zakary Saheb](https://github.com/zaxaheb)

  
## Installation 

after activating your python environment, install required package using this command:

```bash 
  pip install -r requirements.txt
```

    
## Usage/Examples

- Demo page
there is a page where you can find all the demos. you have to do the following command go to the link that appears:

```bash 
  python demo.py
```

here is what it will look like:

![Imgur](https://i.imgur.com/sSxIAzb.png)

- Train model

Your can also train one of our model using :

```bash 
  python train_vxm.py 
  python train_inverse.py
```

your can explore the different configuration for training in `train_vxm.py` and `train_inverse.py`
