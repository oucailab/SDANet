import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


@METRIC_REGISTRY.register()
def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()



################下面是MSD作者的评估函数

from scipy.signal import convolve2d
# 从 skimage.metrics 直接导入新名称的函数
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# ==============================================================================
# 辅助函数
# ==============================================================================

def img_2d_mat(x_true, x_pred):
    """
    # 将三维的多光谱图像转为2位矩阵
    :param x_true: (H, W, C)
    :param x_pred: (H, W, C)
    :return: a matrix which shape is (C, H * W)
    """
    # 确保输入是 float32 类型
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    
    # 检查维度
    if x_true.ndim != 3 or x_pred.ndim != 3:
        raise ValueError("Input images must be 3-dimensional (H, W, C)")
        
    h, w, c = x_true.shape
    
    # 优化：使用 numpy 的 reshape/transpose 替代循环
    # 目标形状 (C, H * W)
    x_mat = x_true.transpose(2, 0, 1).reshape(c, h * w)
    y_mat = x_pred.transpose(2, 0, 1).reshape(c, h * w)
    
    return x_mat, y_mat

# ==============================================================================
# 核心指标函数
# ==============================================================================

# 注意：BasicSR 中的指标函数通常接受两个参数：img1 (输出) 和 img2 (GT)
# 但由于你的某些指标需要额外的参数 (如 ratio, data_range)，我们会在包装函数中处理。


def compare_ergas(x_true, x_pred, ratio):
    """
    Calculate ERGAS, ERGAS offers a global indication of the quality of fused image.The ideal value is 0.
    :param x_true: (H, W, C)
    :param x_pred: (H, W, C)
    :param ratio: 上采样系数
    :return:
    """
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred) # 转换为 (C, H*W)
    sum_ergas = 0
    # 循环 C 个波段
    for i in range(x_true.shape[0]):
        vec_x = x_true[i]
        vec_y = x_pred[i]
        err = vec_x - vec_y
        r_mse = np.mean(np.power(err, 2))
        # 增加一个极小值防止除以零
        tmp = r_mse / (np.mean(vec_x)**2 + 1e-9)
        sum_ergas += tmp
    return (100 / ratio) * np.sqrt(sum_ergas / x_true.shape[0])


def compare_sam(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    h, w, c = x_true.shape
    
    # 将 HWC 转换为 (H*W, C) 以便进行逐像素计算
    x_true_flat = x_true.reshape(-1, c)
    x_pred_flat = x_pred.reshape(-1, c)
    
    # 计算分子 (内积)
    numerator = np.sum(x_true_flat * x_pred_flat, axis=1)
    
    # 计算分母 (范数乘积)
    norm_true = np.linalg.norm(x_true_flat, axis=1)
    norm_pred = np.linalg.norm(x_pred_flat, axis=1)
    denominator = norm_true * norm_pred
    
    # 避免除以零和处理零向量
    valid_pixels = denominator > 1e-9
    
    # 计算余弦值，并限制在 [-1, 1] 范围内以避免 arccos 错误
    cosine = np.zeros_like(numerator)
    cosine[valid_pixels] = numerator[valid_pixels] / denominator[valid_pixels]
    cosine = np.clip(cosine, -1.0, 1.0)
    
    # 计算角度 (弧度)
    sam_rad = np.arccos(cosine[valid_pixels])
    
    if len(sam_rad) == 0:
        return 0.0
        
    # 转换为角度并取平均
    sam_deg = np.mean(sam_rad) * 180.0 / np.pi
    return sam_deg


def compare_corr(x_true, x_pred):
    """
    Calculate the cross correlation between x_pred and x_true.
    求对应波段的相关系数，然后取均值
    CC is a spatial measure.
    """
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred) # 转换为 (C, H*W)
    
    # 逐波段（行）去均值
    x_true_mean = np.mean(x_true, axis=1, keepdims=True)
    x_pred_mean = np.mean(x_pred, axis=1, keepdims=True)
    x_true_centered = x_true - x_true_mean
    x_pred_centered = x_pred - x_pred_mean
    
    # 计算协方差 (分子)
    numerator = np.sum(x_true_centered * x_pred_centered, axis=1)
    
    # 计算标准差乘积 (分母)
    std_true = np.sqrt(np.sum(x_true_centered * x_true_centered, axis=1))
    std_pred = np.sqrt(np.sum(x_pred_centered * x_pred_centered, axis=1))
    denominator = std_true * std_pred
    
    # 避免除以零
    denominator[denominator == 0] = 1e-9
    
    # 相关系数
    corr_coeffs = numerator / denominator
    
    # 返回平均相关系数
    return np.mean(corr_coeffs)


def compare_rmse(x_true, x_pred):
    """
    Calculate Root mean squared error
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    
    # 确保维度匹配
    if x_true.shape != x_pred.shape:
        raise ValueError("Input shapes must match for RMSE calculation.")
        
    # 元素总数
    total_elements = np.prod(x_true.shape)
    
    # MSE
    mse = np.mean((x_true - x_pred)**2)
    
    # RMSE
    rmse = np.sqrt(mse)
    return rmse
    
    # 原始代码的实现：
    # return np.linalg.norm(x_true - x_pred) / (np.sqrt(x_true.shape[0] * x_true.shape[1] * x_true.shape[2]))
    # 原始实现等价于 np.sqrt(np.mean((x_true - x_pred)**2))，即 RMSE，保留原意。


def compare_mpsnr(x_true, x_pred, data_range):
    """
    Calculate Mean Peak Signal-to-Noise Ratio (MPSNR)
    :param x_true: Input image must have three dimension (H, W, C)
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    channels = x_true.shape[2]
    
    total_psnr = [peak_signal_noise_ratio(image_true=x_true[:, :, k], image_test=x_pred[:, :, k], data_range=data_range)
                  for k in range(channels)]

    return np.mean(total_psnr)


def compare_mssim(x_true, x_pred, data_range):
    """
    Calculate Mean Structural Similarity Index (MSSIM)
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    
    # 逐通道计算 SSIM
    mssim = [structural_similarity(im1=x_true[:, :, i], im2=x_pred[:, :, i], data_range=data_range)
            for i in range(x_true.shape[2])]

    return np.mean(mssim)

# ------------------------------------------------------------------------------
# 以下指标在 BasicSR 中可能不常用，但我们仍将其定义，方便使用
# ------------------------------------------------------------------------------

def compare_sid(x_true, x_pred):
    """
    SID is an information theoretic measure for spectral similarity and discriminability.
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    N = x_true.shape[2]
    
    # 避免 log(0)
    x_true_safe = x_true + 1e-9
    x_pred_safe = x_pred + 1e-9
    
    sid_per_band = []
    for i in range(N):
        p = x_pred[:, :, i].ravel()
        q = x_true[:, :, i].ravel()
        p_safe = x_pred_safe[:, :, i].ravel()
        q_safe = x_true_safe[:, :, i].ravel()
        
        # SID = sum(p * log(p/q)) + sum(q * log(q/p))
        term1 = np.sum(p * np.log10(p_safe / q_safe))
        term2 = np.sum(q * np.log10(q_safe / p_safe))
        
        sid_per_band.append(term1 + term2)
        
    # 原始代码是除以像素数，然后取平均
    pixel_count = x_true.shape[0] * x_true.shape[1]
    return np.mean(np.array(sid_per_band) / pixel_count)


def compare_appsa(x_true, x_pred):
    """
    Angular Power Spectrum Similarity (APPSA)
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    
    # 确保 HWC 格式
    if x_true.ndim != 3:
        raise ValueError("Input images must be 3-dimensional (H, W, C)")
        
    # 沿 C 维度计算内积
    nom = np.sum(x_true * x_pred, axis=2)
    
    # 沿 C 维度计算范数
    denom = np.linalg.norm(x_true, axis=2) * np.linalg.norm(x_pred, axis=2)

    # 计算余弦值
    # 避免除以零
    cos_val = nom / (denom + 1e-9)
    
    # 限制在 [-1, 1] 范围内
    cos = np.clip(cos_val, -1.0, 1.0)
    
    # 计算角度 (弧度)
    appsa = np.arccos(cos)
    
    # 返回平均角度
    return np.mean(appsa)


def compare_mare(x_true, x_pred):
    """
    Mean Absolute Relative Error (MARE)
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    diff = x_true - x_pred
    abs_diff = np.abs(diff)
    
    # added epsilon to avoid division by zero.
    relative_abs_diff = np.divide(abs_diff, x_true + 1e-9) 
    
    return np.mean(relative_abs_diff)


def img_qi(img1, img2, block_size=8):
    """
    Calculate Q-Index for a single band.
    """
    N = block_size ** 2
    sum2_filter = np.ones((block_size, block_size))

    img1_sq = img1 * img1
    img2_sq = img2 * img2
    img12 = img1 * img2

    # 使用 'valid' 模式进行卷积
    img1_sum = convolve2d(img1, sum2_filter, mode='valid')
    img2_sum = convolve2d(img2, sum2_filter, mode='valid')
    img1_sq_sum = convolve2d(img1_sq, sum2_filter, mode='valid')
    img2_sq_sum = convolve2d(img2_sq, sum2_filter, mode='valid')
    img12_sum = convolve2d(img12, sum2_filter, mode='valid')

    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum * img1_sum + img2_sum * img2_sum
    numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul
    
    quality_map = np.ones(denominator.shape)
    
    # 处理特殊情况 1: denominator1 == 0 且 img12_sq_sum_mul != 0
    index1 = (denominator1 == 0) & (img12_sq_sum_mul != 0)
    quality_map[index1] = 2 * img12_sum_mul[index1] / img12_sq_sum_mul[index1]
    
    # 处理一般情况: denominator != 0
    index2 = (denominator != 0)
    quality_map[index2] = numerator[index2] / denominator[index2]
    
    return quality_map.mean()


def compare_qave(x_true, x_pred, block_size=8):
    """
    Calculate Average Q-Index (QAVE)
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    n_bands = x_true.shape[2]
    q_orig = np.zeros(n_bands)
    for idim in range(n_bands):
        q_orig[idim] = img_qi(x_true[:, :, idim], x_pred[:, :, idim], block_size)
    return q_orig.mean()


# ==============================================================================
# BasicSR 包装函数 (使用注册机制)
# ==============================================================================

# 导入 BasicSR 的注册函数
try:
    from basicsr.utils.registry import METRIC_REGISTRY
except ImportError:
    # 如果没有找到注册机制，则定义一个简单的占位符
    print("Warning: BasicSR METRIC_REGISTRY not found. Using placeholder.")
    class Registry:
        def __init__(self, name): self._name = name
        def register(self, func=None, name=None):
            def wrapper(f): return f
            return wrapper if func is None else func
    METRIC_REGISTRY = Registry('metric')


# ------------------------------------------------------------------------------
# 1. 包装 MPSNR
# ------------------------------------------------------------------------------
@METRIC_REGISTRY.register()
def calculate_mpsnr_hsi(img, img2, crop_border, test_y_channel=False, data_range=1.0):
    """
    BasicSR metric wrapper for compare_mpsnr.
    img, img2: 输入的 NumPy 数组 (HWC, float32)
    """
    # BasicSR 默认输入是 HWC 的 NumPy 数组 (0-255 或 0-1)
    # 确保是 float32
    img, img2 = img.astype(np.float32), img2.astype(np.float32)
    
    # 裁剪边界 (如果需要)
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        
    # 忽略 test_y_channel，因为高光谱图像通常评估所有波段
    
    return compare_mpsnr(x_true=img2, x_pred=img, data_range=data_range)


# ------------------------------------------------------------------------------
# 2. 包装 MSSIM
# ------------------------------------------------------------------------------
@METRIC_REGISTRY.register()
def calculate_mssim_hsi(img, img2, crop_border, test_y_channel=False, data_range=1.0):
    """
    BasicSR metric wrapper for compare_mssim.
    """
    img, img2 = img.astype(np.float32), img2.astype(np.float32)
    
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        
    return compare_mssim(x_true=img2, x_pred=img, data_range=data_range)


# ------------------------------------------------------------------------------
# 3. 包装 ERGAS
# ------------------------------------------------------------------------------
@METRIC_REGISTRY.register()
def calculate_ergas_hsi(img, img2, crop_border, test_y_channel=False, ratio=4):
    """
    BasicSR metric wrapper for compare_ergas.
    需要 ratio 参数，通常在配置文件中设置。
    """
    img, img2 = img.astype(np.float32), img2.astype(np.float32)
    
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        
    return compare_ergas(x_true=img2, x_pred=img, ratio=ratio)


# ------------------------------------------------------------------------------
# 4. 包装 SAM
# ------------------------------------------------------------------------------
@METRIC_REGISTRY.register()
def calculate_sam_hsi(img, img2, crop_border, test_y_channel=False):
    """
    BasicSR metric wrapper for compare_sam.
    """
    img, img2 = img.astype(np.float32), img2.astype(np.float32)
    
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        
    return compare_sam(x_true=img2, x_pred=img)


# ------------------------------------------------------------------------------
# 5. 包装 CrossCorrelation (CC)
# ------------------------------------------------------------------------------
@METRIC_REGISTRY.register()
def calculate_corr_hsi(img, img2, crop_border, test_y_channel=False):
    """
    BasicSR metric wrapper for compare_corr.
    """
    img, img2 = img.astype(np.float32), img2.astype(np.float32)
    
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        
    return compare_corr(x_true=img2, x_pred=img)


# ------------------------------------------------------------------------------
# 6. 包装 RMSE
# ------------------------------------------------------------------------------
@METRIC_REGISTRY.register()
def calculate_rmse_hsi(img, img2, crop_border, test_y_channel=False):
    """
    BasicSR metric wrapper for compare_rmse.
    """
    img, img2 = img.astype(np.float32), img2.astype(np.float32)
    
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        
    return compare_rmse(x_true=img2, x_pred=img)

# ------------------------------------------------------------------------------
# 7. 包装 APPSA
# ------------------------------------------------------------------------------
@METRIC_REGISTRY.register()
def calculate_appsa_hsi(img, img2, crop_border, test_y_channel=False):
    """
    BasicSR metric wrapper for compare_appsa.
    """
    img, img2 = img.astype(np.float32), img2.astype(np.float32)
    
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        
    return compare_appsa(x_true=img2, x_pred=img)

# ------------------------------------------------------------------------------
# 8. 包装 QAVE
# ------------------------------------------------------------------------------
@METRIC_REGISTRY.register()
def calculate_qave_hsi(img, img2, crop_border, test_y_channel=False, block_size=8):
    """
    BasicSR metric wrapper for compare_qave.
    """
    img, img2 = img.astype(np.float32), img2.astype(np.float32)
    
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        
    return compare_qave(x_true=img2, x_pred=img, block_size=block_size)

@METRIC_REGISTRY.register()
def calculate_psnr_float(img, img2, crop_border, test_y_channel=False, data_range=1.0, **kwargs):
    """为 [0, 1] 范围的浮点图像计算 PSNR."""
    img, img2 = img.astype(np.float32), img2.astype(np.float32)
    
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    # 使用 skimage 的函数，并传入正确的 data_range
    return peak_signal_noise_ratio(image_true=img2, image_test=img, data_range=data_range)


# 新增一个支持 data_range 的 SSIM 函数
@METRIC_REGISTRY.register()
def calculate_ssim_float(img, img2, crop_border, test_y_channel=False, data_range=1.0, **kwargs):
    """为 [0, 1] 范围的浮点图像计算 SSIM."""
    img, img2 = img.astype(np.float32), img2.astype(np.float32)
    
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    # multichannel=True 会为所有通道计算 SSIM 并返回平均值，适合高光谱
    return structural_similarity(im1=img, im2=img2, multichannel=True, data_range=data_range)