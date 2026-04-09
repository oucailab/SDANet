import numpy as np
import scipy.io as sio
import torch
from torch.utils import data as data

from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class TestMatDatasetCHU(data.Dataset):
    """Validation dataset that reads all HSI samples from one .mat file."""

    def __init__(self, opt):
        super(TestMatDatasetCHU, self).__init__()
        self.opt = opt
        test_mat_path = opt['dataroot_gt']

        all_data = sio.loadmat(test_mat_path)
        self.all_gt = np.array(all_data['gt'], dtype=np.float32)
        self.all_lq = np.array(all_data['ms'], dtype=np.float32)
        self.all_lms = np.array(all_data['ms_bicubic'], dtype=np.float32)
        self.num_images = self.all_gt.shape[0]

    def __getitem__(self, index):
        img_gt = self.all_gt[index]
        img_lq = self.all_lq[index]
        img_lms = self.all_lms[index]

        img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float()
        img_lq = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq, (2, 0, 1)))).float()
        img_lms = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lms, (2, 0, 1)))).float()

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lms': img_lms,
            'lq_path': f'test_img_{index}',
            'gt_path': f'test_img_{index}'
        }

    def __len__(self):
        return self.num_images