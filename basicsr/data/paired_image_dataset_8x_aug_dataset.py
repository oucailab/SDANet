import numpy as np
import os
import scipy.io as sio
import torch
from torch.utils import data as data

from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY

def data_augmentation(image, mode=0):
    """
    Performs 8-way data augmentation.
    
    Args:
        image (np.array): Input image as a NumPy array.
        mode (int): The augmentation mode (0 to 7).
    """
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(image))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(image, k=3))

@DATASET_REGISTRY.register()
class PairedMatDataset8xAug(data.Dataset):
    """Paired .mat dataset for hyperspectral image restoration.

    Reads LQ (ms), LMS (ms_bicubic), and GT (gt) from a single .mat file.
    Implements a systematic 8-way data augmentation to replicate HSRMamba/MSDformer strategy.
    """

    def __init__(self, opt):
        super(PairedMatDataset8xAug, self).__init__()
        self.opt = opt
        
        self.mat_folder = opt['dataroot_gt']
        self.mat_files = sorted([os.path.join(self.mat_folder, f) for f in os.listdir(self.mat_folder) if f.endswith('.mat')])

    def __getitem__(self, index):
        # Calculate the original image index and the augmentation mode
        original_index = index // 8
        augmentation_mode = index % 8

        mat_path = self.mat_files[original_index]
        try:
            data = sio.loadmat(mat_path)
        except Exception as e:
            logger = get_root_logger()
            logger.warning(f'Failed to load .mat file {mat_path}, skipping. Error: {e}')
            return self.__getitem__((index + 1) % (len(self.mat_files) * 8))

        img_gt = np.array(data['gt'], dtype=np.float32)
        img_lq = np.array(data['ms'], dtype=np.float32)
        img_lms = np.array(data['ms_bicubic'], dtype=np.float32)
        
        scale = self.opt['scale']

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            h_gt, w_gt, _ = img_gt.shape
            
            # 1. Randomly select top-left coordinates on the GT image
            top = np.random.randint(0, h_gt - gt_size + 1)
            left = np.random.randint(0, w_gt - gt_size + 1)

            # 2. Crop GT and LMS
            img_gt = img_gt[top:top + gt_size, left:left + gt_size, :]
            img_lms = img_lms[top:top + gt_size, left:left + gt_size, :]

            # 3. Calculate corresponding coordinates for LQ and crop
            top_lq, left_lq = int(top // scale), int(left // scale)
            lq_size = int(gt_size // scale)
            img_lq = img_lq[top_lq:top_lq + lq_size, left_lq:left_lq + lq_size, :]

            # Apply the systematic 8-way data augmentation
            # Note: We ensure memory contiguity before transposing later
            img_gt = data_augmentation(img_gt, mode=augmentation_mode)
            img_lq = data_augmentation(img_lq, mode=augmentation_mode)
            img_lms = data_augmentation(img_lms, mode=augmentation_mode)
        
        # Convert to PyTorch CHW format
        # np.ascontiguousarray is important for safe tensor conversion
        img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float()
        img_lq = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq, (2, 0, 1)))).float()
        img_lms = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lms, (2, 0, 1)))).float()

        return {'lq': img_lq, 'gt': img_gt, 'lms': img_lms, 'lq_path': mat_path, 'gt_path': mat_path}

    def __len__(self):
        # The dataset is 8 times larger due to the augmentation
        return len(self.mat_files) * 8