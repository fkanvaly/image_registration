import sys

sys.path.append("./")
from scripts.demo import mutliapp, demo__intro, demo__experiment, demo__voxelmorph, demo__inverse, demo__mnist_inv, demo__mnist_vxm

app = mutliapp.MultiApp()
app.add_app("Introduction", demo__intro.app)
app.add_app("VoxelMorph", demo__voxelmorph.app)
app.add_app("MNIST - VoxelMorph", demo__mnist_vxm.app)
app.add_app("Inverse-Consistent", demo__inverse.app)
app.add_app("MNIST - Inverse-Consistent", demo__mnist_inv.app)
app.add_app("Brain Images", demo__experiment.app)
app.run()