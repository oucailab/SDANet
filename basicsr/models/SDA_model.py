import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY, LOSS_REGISTRY


@MODEL_REGISTRY.register()
class SDAModel(SRModel):
    """
    A model wrapper for SDA architecture.

    This model inherits directly from SRModel, leveraging its robust
    training and validation pipeline for single-image super-resolution.
    The specific network architecture (SDA) will be instantiated
    based on the provided configuration file.
    """
    pass