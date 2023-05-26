#!/usr/bin/env python

"""

Purpose :

"""

# from torchmetrics.functional import structural_similarity_index_measure
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from scipy.stats import norm

# from pytorch_msssim import SSIM
from .perceptual_loss import PerceptualLoss

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class SoftNCutsLoss(nn.Module):
    r"""Implementation of the continuous N-Cut loss, as in:
    'W-Net: A Deep Model for Fully Unsupervised Image Segmentation', by Xia, Kulis (2017)"""

    def __init__(self, radius=4, sigma_x=5, sigma_i=1):
        r"""
        :param radius: Radius of the spatial interaction term
        :param sigma_x: Standard deviation of the spatial Gaussian interaction
        :param sigma_i: Standard deviation of the pixel value Gaussian interaction
        """
        super(SoftNCutsLoss, self).__init__()
        self.radius = radius
        self.sigma_x = sigma_x  # Spatial standard deviation
        self.sigma_i = sigma_i  # Pixel value standard deviation

    def gaussian_kernel3d(self):
        neighborhood_size = 2 * self.radius + 1
        voxel_neighborhood = np.linspace(-self.radius, self.radius, neighborhood_size) ** 2
        xy, yz, zx = np.meshgrid(voxel_neighborhood, voxel_neighborhood, voxel_neighborhood)
        dist = (xy + yz + zx) / self.sigma_x
        kernel = norm.pdf(dist) / norm.pdf(0)
        kernel = torch.from_numpy(kernel.astype(np.float32))
        kernel = kernel.view((1, 1, kernel.shape[0], kernel.shape[1], kernel.shape[2]))
        kernel = kernel.cuda()
        return kernel

    def forward(self, inputs, labels):
        r"""Computes the continuous N-Cut loss, given a set of class probabilities (labels) and raw images (inputs).
        Small modifications have been made here for efficiency -- specifically, we compute the pixel-wise weights
        relative to the class-wide average, rather than for every individual pixel.
        :param labels: Predicted class probabilities
        :param inputs: Raw images
        :return: Continuous N-Cut loss
        """
        num_classes = labels.shape[1]
        loss = 0
        kernel = self.gaussian_kernel3d()

        for k in range(num_classes):
            # Compute the average pixel value for this class, and the difference from each pixel
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(inputs * class_probs, dim=(2, 3, 4), keepdim=True) / \
                torch.add(torch.mean(class_probs, dim=(2, 3, 4), keepdim=True), 1e-5)
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)

            # Weight the loss by the difference from the class average.
            weights = torch.exp(diff.pow(2).mul(-1 / self.sigma_i ** 2))

            # Compute N-cut loss, using the computed weights matrix, and a Gaussian spatial filter
            numerator = torch.sum(class_probs * F.conv3d(class_probs * weights, kernel, padding=self.radius))
            denominator = torch.sum(class_probs * F.conv3d(weights, kernel, padding=self.radius))
            loss += nn.L1Loss()(numerator / torch.add(denominator, 1e-6), torch.zeros_like(numerator))

        return num_classes - loss


class ReconstructionLoss(nn.Module):
    def __init__(self, recr_loss_model_path=None, loss_type="L1"):
        super(ReconstructionLoss, self).__init__()
        self.loss = PerceptualLoss(loss_model="unet3Dds", model_load_path=recr_loss_model_path, loss_type=loss_type)

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


def l2_regularisation_loss(model):
    l2_reg = torch.tensor(0.0, requires_grad=True)

    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_reg = l2_reg + param.norm(2)
    return l2_reg

class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, res_map_shape, class_assignments):
        return self.loss(res_map_shape, class_assignments)


class ContinuityLoss(nn.Module):
    def __init__(self, batch_size=15, patch_size=32, num_classes=6):
        super(ContinuityLoss, self).__init__()
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.cont_width_target = torch.zeros(
            (batch_size, self.num_classes, patch_size - 1, patch_size, patch_size)).float().cuda()
        self.cont_length_target = torch.zeros(
            (batch_size, self.num_classes, patch_size, patch_size - 1, patch_size)).float().cuda()
        self.cont_depth_target = torch.zeros(
            (batch_size, self.num_classes, patch_size, patch_size, patch_size - 1)).float().cuda()
        self.loss = torch.nn.L1Loss()

    def forward(self, class_probs):
        cont_width_op = class_probs[:, :, 1:, :, :] - class_probs[:, :, 0:-1, :, :]
        cont_length_op = class_probs[:, :, :, 1:, :] - class_probs[:, :, :, 0:-1, :]
        cont_depth_op = class_probs[:, :, :, :, 1:] - class_probs[:, :, :, :, 0:-1]
        continuity_loss_width = self.loss(cont_width_op,
                                          self.cont_width_target[:class_probs.shape[0], :, :, :, :].clone())
        continuity_loss_length = self.loss(cont_length_op,
                                           self.cont_length_target[:class_probs.shape[0], :, :, :, :].clone())
        continuity_loss_depth = self.loss(cont_depth_op,
                                          self.cont_depth_target[:class_probs.shape[0], :, :, :, :].clone())
        return continuity_loss_width + continuity_loss_length + continuity_loss_depth
