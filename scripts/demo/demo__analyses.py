import streamlit as st
import sys

sys.path.append("./")


def app():
    st.write('<style>img{max-width:1000px}</style>', unsafe_allow_html=True)
    st.write(r"""
# ANALYSES AND RESULTS

## VoxelMorph - MNIST
### 1. $\lambda = 0$
![test](https://i.imgur.com/pcJ6jzD.png)

Flow:

![flow](https://i.imgur.com/Y3zDgJh.png)

- Dice score : 0.926
- Injectivity indicator : $51.4\%$

Here we see that without any constraint on smoothness we have a good dice score, but the flow is complicated and the jacobian determinant has several pixels (the brown ones) which are close to zero. It shows that it is not very diffeomorphic and we want it to be diffeomorphic because it is important on brain images.

### 2. $\lambda = 0.5$
![test](https://i.imgur.com/GokdrmY.png)

Flow : 

![flow](https://i.imgur.com/CBeralz.png)

- Dice score : 0.789
- Injectivity indicator : $85.7\%$

We see that with regularization on the smoothness the flow is more controlled and we don't have any pixel which is not "diffeomorphic". The injectivity inducator is much higher. Therefore the registration is more complicated and the moved image doesn't look like the fixed one but on brain images the images won't so different. Let's see with similar images :

### 3. $\lambda = 0.5$,  same numbers
![test](https://i.imgur.com/Kc2nEFq.png)

Flow : 

![flow](https://i.imgur.com/VEuTktf.png)

- Dice score : 0.785

We can see that the moved number is not exactly the same that the fixed one but they are aligned and well registrated

_______________________________________________________________________________________
## Inverse-Consistent  - MNIST
### 1. With default parameters
Parameters : We took the coefficients used in the article 
- Inverse coefficient : 0.05
- Smoothness coefficient : 0.5
- Antifold coefficient : 100000

Injectivity indicator : $90.2\%$

Some examples :

![test](https://i.imgur.com/ECBYh5C.png)
![flow](https://i.imgur.com/r4Ij71I.png)

Dice score : 0.819

![test](https://i.imgur.com/arIXsJ0.png)
![flow](https://i.imgur.com/yBhnL2B.png)

Dice score : 0.789


Losses $\times$ their coefficients :
![losses](https://i.imgur.com/1x63ZWH.png)

It seems that the similarity loss (mse on the images) is the closest to the total loss. But the new losses are important too.
Let's introduce the role of each constraint thanks to an ablation study.
### 2. Ablation Study
    
#### Smooth_loss ablation : 
    
Injectivity indicator : $67.7\%$
    
Same number and same topology

![test](https://i.imgur.com/OGLXdIA.png)
![flow](https://i.imgur.com/Chga6ps.png)
    
   - DICE score : 0.801

Different number and different topology

![test](https://i.imgur.com/3Q7E3u0.png)
![flow](https://i.imgur.com/j4A1lNI.png)
    
   - DICE score 0.778
    
We can see that the flow is less smooth here but still controled (it shows that there is another constraint which has influence on that, it is actually the antifold constraint). However the injectivity indicator is very lower than with the constraint. Finally smoothness has a direct impact on the injectivity so on the bijectivity too. It confirms what we saw with VoxelMorph.
    
#### AntiFold_loss ablation : 

Injectivity indicator : $80.2\%$
    
Same number and same topology

![test](https://i.imgur.com/9sa098E.png)
![flow](https://i.imgur.com/2sykhdd.png)
    
   - DICE score : 0.790

Different number and different topology

![test](https://i.imgur.com/UG4GQtS.png)
![flow](https://i.imgur.com/nBqWzJ4.png)

   - DICE score : 0.798
    
The antifold constraint is supposed to be very important since in the article they use a huge coefficient. The percentage of injectivity of the points is lower without this loss, it has an impact on the bijectivity, but less than the smoothness constraint. However we can see an area where it isn't diffeomorphic with the jacobian determinant, so this constraint must have an impact on the diffeomorphism but in a different way from the smoothness. In the article they use a big coefficient of regularization because on brain images, the registration must not be a "fold", it wouldn't make any sense. Moreover the dice score does'nt seem to change so it has not an influence on that (like the smoothness)
    
#### Inverse_loss ablation : 
   
Injectivity indicator : $90.1\%$

Same number and same topology

![test](https://i.imgur.com/cPi9IQ6.png)
![flow](https://i.imgur.com/vHP2ffZ.png)
    
    DICE score : 0.789

Different number and different topology

![test](https://i.imgur.com/90GKjTD.png)
![flow](https://i.imgur.com/CaSsr2c.png)
    
    DICE score : 0.777



Finally we can see that with this constraint we have a better score, however it is very light, maybe because the coefficient used in the article is the smallest (0.05) in comparaison to the others.

______________________________________________________________
# On brain images

## VoxelMorph
- Without smoothness

![test](https://i.imgur.com/vJWU7qx.png)

Dice score 0.970

We can see that some areas have disatppeared, whereas it shouldn't. And the jacobian determinant can shows that, it is not diffeomorphic at all, there are flow directions which intercept each other (different color for different orientations) and it shows discontinuities.



- With the constraint

![test](https://i.imgur.com/8TFwnwZ.png)

Dice score : 0.970

Here we can see that the constraints are well adapted, there is a small registration (little arrow on the flow image), but the dice score is good! It is non-linear since the arrows don't go in the same way in the flow. 
The amplitude of the flow is very lower than without the constraint and there is less discontinuity 

## Inverse 
- With the constraints

![test](https://i.imgur.com/QnQhHLB.png)
Dice score : 0.969

We can see a difference in the flows, this one is more homogeneous in amplitude than with voxelmorph, there is less discontinuity, certainly because there is the antifold constraint and it results in a lower score (which is not necessary a bad thing here)

____________________________________
# Conclusion 

The second model is more complex, there is not only one constraint for the diffeomorphism, there is the antifold constraint which is important on brain image but the two are complementary and don't work the same way.  Moreover the inverse constraint accentuates the strength of the similarity between the moved image and the fixed one.

Of course, even if VoxelMorph is simpler, the smoothness constraint works really good and avoid to have bad registration when there are different topologies.
    
""")


if __name__ == '__main__':
    app()
