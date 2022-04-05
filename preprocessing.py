import os, sys
# sys.path.append('/home/check5657/Workspace/pycharm_pytorch')
import mh_overlay, mh_loader
from network.resunet_3d import *
import segmentation_models_pytorch as smp

import pytorch3dunet.unet3d.model as p
import torch
import pytorch_model_summary as su

model=p.ResidualUNet3D(in_channels=2, out_channels=2)
print(model)

print(su.summary(model,torch.zeros((16,2,16,128,128)), show_parent_layers=True))


model2d=smp.UnetPlusPlus(in_channels=2)
print(model2d)



