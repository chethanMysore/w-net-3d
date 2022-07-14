#!/usr/bin/env python

"""

Purpose :

"""

import torch
import torch.nn as nn
import torch.utils.data
from torchmetrics import StructuralSimilarityIndexMeasure

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class SoftNCutsLoss(nn.Module):
    def __init__(self, depth, length, width, std_position=1):
        super(SoftNCutsLoss, self).__init__()
        meshgrid_x, meshgrid_y, meshgrid_z = torch.meshgrid(torch.arange(0, depth, dtype=float),
                                                            torch.arange(0, length, dtype=float),
                                                            torch.arange(0, width, dtype=float))
        meshgrid_x = torch.reshape(meshgrid_x, (length * width * depth,))
        meshgrid_y = torch.reshape(meshgrid_y, (length * width * depth,))
        meshgrid_z = torch.reshape(meshgrid_z, (length * width * depth,))
        A_x = SoftNCutsLoss._outer_product(meshgrid_x, torch.ones(meshgrid_x.size(), dtype=meshgrid_x.dtype, device=meshgrid_x.device))
        A_y = SoftNCutsLoss._outer_product(meshgrid_y, torch.ones(meshgrid_y.size(), dtype=meshgrid_y.dtype, device=meshgrid_y.device))
        A_z = SoftNCutsLoss._outer_product(meshgrid_z, torch.ones(meshgrid_z.size(), dtype=meshgrid_z.dtype, device=meshgrid_z.device))

        del meshgrid_x, meshgrid_y, meshgrid_z

        xi_xj = A_x - A_x.permute(1, 0)
        yi_yj = A_y - A_y.permute(1, 0)
        zi_zj = A_z - A_z.permute(1, 0)

        sq_distance_matrix = torch.square(xi_xj) + torch.square(yi_yj) + torch.square(zi_zj)

        del A_x, A_y, A_z, xi_xj, yi_yj, zi_zj

        self.dist_weight = torch.exp(-torch.divide(sq_distance_matrix, torch.square(torch.tensor(std_position))))

    @staticmethod
    def _outer_product(v1, v2):
        """
        Inputs:
        v1 : m*1 tf array
        v2 : m*1 tf array
        Output :
        v1 x v2 : m*m array
        """
        v1 = torch.reshape(v1, (-1,))
        v2 = torch.reshape(v2, (-1,))
        v1 = torch.unsqueeze(v1, 0)
        v2 = torch.unsqueeze(v2, 0)
        return torch.matmul(v1.T, v2)

    def _edge_weights(self, flatten_patch, std_intensity=3):
        """
        Inputs :
        flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels)
        std_intensity : standard deviation for intensity
        std_position : standard devistion for position
        radius : the length of the around the pixel where the weights
        is non-zero
        rows : rows of the original image (unflattened image)
        cols : cols of the original image (unflattened image)
        Output :
        weights :  2d tf array edge weights in the pixel graph
        Used parameters :
        n : number of pixels
        """
        A = SoftNCutsLoss._outer_product(flatten_patch, torch.ones_like(flatten_patch))
        intensity_weight = torch.exp(-1 * torch.square((torch.divide((A - A.T), std_intensity)))).detach().cpu()
        del A
        return torch.multiply(intensity_weight, self.dist_weight)

    @staticmethod
    def _numerator(k_class_prob, weights):
        """
        Inputs :
        k_class_prob : k_class pixelwise probability (rows*cols) tensor
        weights : edge weights n*n tensor
        """
        k_class_prob = torch.reshape(k_class_prob, (-1,))
        return torch.sum(torch.multiply(weights, SoftNCutsLoss._outer_product(k_class_prob, k_class_prob)))

    @staticmethod
    def _denominator(k_class_prob, weights):
        """
        Inputs:
        k_class_prob : k_class pixelwise probability (rows*cols) tensor
        weights : edge weights	n*n tensor
        """
        k_class_prob = torch.reshape(k_class_prob, (-1,))
        return torch.sum(torch.multiply(weights, SoftNCutsLoss._outer_product(k_class_prob,
                                                                              torch.ones(k_class_prob.size(),
                                                                                         dtype=k_class_prob.dtype,
                                                                                         layout=k_class_prob.layout,
                                                                                         device=k_class_prob.device))))

    def forward(self, patch, prob, k):
        """
        Inputs:
        prob : (rows*cols*k) tensor
        k : number of classes (integer)
        flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels)
        rows : number of the rows in the original image
        cols : number of the cols in the original image
        Output :
        soft_n_cut_loss tensor for a single image
        """
        flatten_patch = torch.flatten(patch)
        soft_n_cut_loss = k
        weights = self._edge_weights(flatten_patch)
        prob = prob.cpu()

        for t in range(k):
            soft_n_cut_loss = soft_n_cut_loss - (
                    SoftNCutsLoss._numerator(prob[:, :, :, t], weights) / SoftNCutsLoss._denominator(prob[:, :, :, t],
                                                                                                     weights))

        del weights
        del flatten_patch
        return soft_n_cut_loss.float().cuda()


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, y_pred, y_true):
        return 1 - self.ssim(y_pred, y_true)
