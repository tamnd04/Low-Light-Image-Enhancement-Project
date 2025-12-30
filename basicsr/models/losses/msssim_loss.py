"""
MS-SSIM Loss Implementation
Based on: "Multiscale structural similarity for image quality assessment" (Wang et al., 2003)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size, sigma):
    """Create a 1D Gaussian kernel"""
    gauss = torch.Tensor([
        torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create a 2D Gaussian window"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Compute SSIM between two images"""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class MS_SSIM(nn.Module):
    """Multi-Scale Structural Similarity Index"""
    
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            self.window = self.window.type_as(img1)

        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)
        levels = weights.size(0)
        mssim = []
        mcs = []

        for i in range(levels):
            ssim_val, cs = self._ssim_per_channel(img1, img2, self.window, self.window_size, self.channel)
            mssim.append(ssim_val)
            mcs.append(cs)

            if i < levels - 1:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # Calculate the overall MS-SSIM
        pow1 = mcs ** weights
        pow2 = mssim ** weights
        output = torch.prod(pow1[:-1]) * pow2[-1]

        return output

    def _ssim_per_channel(self, img1, img2, window, window_size, channel):
        """Calculate SSIM and contrast-structure for each channel"""
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

        if self.size_average:
            ssim_val = ssim_map.mean()
            cs = cs_map.mean()
        else:
            ssim_val = ssim_map.mean(1).mean(1).mean(1)
            cs = cs_map.mean(1).mean(1).mean(1)

        return ssim_val, cs


class MS_SSIMLoss(nn.Module):
    """MS-SSIM Loss for training (returns 1 - MS_SSIM)"""
    
    def __init__(self, window_size=11, size_average=True, channel=3, data_range=1.0, loss_weight=1.0):
        super(MS_SSIMLoss, self).__init__()
        self.ms_ssim = MS_SSIM(window_size, size_average, channel)
        self.data_range = data_range
        self.loss_weight = loss_weight

    def forward(self, img1, img2):
        ms_ssim_val = self.ms_ssim(img1, img2)
        loss = 1 - ms_ssim_val
        return self.loss_weight * loss
