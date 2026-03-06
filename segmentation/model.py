import os
import torch
from torch import nn
from torch.utils import checkpoint

from mmengine.model import BaseModule
from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

build = import_abspy(
    "models", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)


Backbone_VSSM: nn.Module = build.vmamba.Backbone_VSSM

@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_VSSM(BaseModule, Backbone_VSSM):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_VSSM.__init__(self, *args, **kwargs)


# ======== EQVMamba ======== #
Backbone_EQVSSM: nn.Module = build.eq_vmamba_light.Backbone_EQVSSM

@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_EQVSSM(BaseModule, Backbone_EQVSSM):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_EQVSSM.__init__(self, *args, **kwargs)



# ======== EQVMambaBconv ======== #
Backbone_EQVSSMBconv: nn.Module = build.eq_vmamba_Bconv_light.Backbone_EQVSSMBconv

@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_EQVSSMBconv(BaseModule, Backbone_EQVSSMBconv):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_EQVSSMBconv.__init__(self, *args, **kwargs)




