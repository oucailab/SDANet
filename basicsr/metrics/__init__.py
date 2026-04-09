from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .psnr_ssim import (calculate_corr_hsi, calculate_ergas_hsi,
                        calculate_mpsnr_hsi, calculate_psnr,
                        calculate_sam_hsi, calculate_ssim,
                        calculate_ssim_float)

__all__ = [
    'calculate_psnr',
    'calculate_ssim',
    'calculate_ssim_float',
    'calculate_mpsnr_hsi',
    'calculate_sam_hsi',
    'calculate_ergas_hsi',
    'calculate_corr_hsi',
]


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
