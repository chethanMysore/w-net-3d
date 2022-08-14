#!/usr/bin/env python

"""

Purpose :

"""

import torch
import torch.nn as nn
import torch.utils.data
from torchmetrics import StructuralSimilarityIndexMeasure
import numpy as np
import torchio as tio

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class SoftNCutsLoss_v1(nn.Module):
    def __init__(self, depth, length, width, std_position=1):
        super(SoftNCutsLoss, self).__init__()
        meshgrid_x, meshgrid_y, meshgrid_z = torch.meshgrid(torch.arange(0, depth, dtype=float),
                                                            torch.arange(0, length, dtype=float),
                                                            torch.arange(0, width, dtype=float))
        meshgrid_x = torch.reshape(meshgrid_x, (length * width * depth,))
        meshgrid_y = torch.reshape(meshgrid_y, (length * width * depth,))
        meshgrid_z = torch.reshape(meshgrid_z, (length * width * depth,))
        A_x = SoftNCutsLoss._outer_product(meshgrid_x, torch.ones(meshgrid_x.size(), dtype=meshgrid_x.dtype,
                                                                  device=meshgrid_x.device))
        A_y = SoftNCutsLoss._outer_product(meshgrid_y, torch.ones(meshgrid_y.size(), dtype=meshgrid_y.dtype,
                                                                  device=meshgrid_y.device))
        A_z = SoftNCutsLoss._outer_product(meshgrid_z, torch.ones(meshgrid_z.size(), dtype=meshgrid_z.dtype,
                                                                  device=meshgrid_z.device))

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


class SoftNCutsLoss(nn.Module):
    def __init__(self, radius=4, sigmaI=10, sigmaX=4, num_classes=8, batch_size=15, patch_size=32):
        super(SoftNCutsLoss, self).__init__()
        self.radius = radius
        self.sigmaI = sigmaI
        self.sigmaX = sigmaX
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.ip_shape = (batch_size, 1, patch_size, patch_size, patch_size)
        self.pad = torch.nn.ConstantPad3d(radius - 1, np.finfo(np.float).eps)
        self.dissim_matrix = torch.zeros(
            (self.ip_shape[0], self.ip_shape[1], self.ip_shape[2], self.ip_shape[3], self.ip_shape[4],
             (radius - 1) * 2 + 1,
             (radius - 1) * 2 + 1, (radius - 1) * 2 + 1)).cuda()
        self.dist = torch.zeros((2 * (self.radius - 1) + 1, 2 * (self.radius - 1) + 1, 2 * (self.radius - 1) + 1)).cuda()
        self.cropped_seg = torch.zeros((batch_size, num_classes, patch_size, patch_size, patch_size,
                                        2 * (self.radius - 1) + 1, 2 * (self.radius - 1) + 1,
                                        2 * (self.radius - 1) + 1)).cuda()

    def _cal_weights(self, batch, padded_batch):
        """
        Inputs:
        batch : ip batch (B x C x D x H x W)
        padded_batch : padded ip batch
        Output :
        weight and sum weight
        """
        # According to the weight formula, when Euclidean distance < r,the weight is 0,
        # so reduce the dissim matrix size to radius-1 to save time and space.
        # print("calculating weights.")
        temp_dissim = self.dissim_matrix[:batch.shape[0], :, :, :, :, :, :, :]
        for x in range(2 * (self.radius - 1) + 1):
            for y in range(2 * (self.radius - 1) + 1):
                for z in range(2 * (self.radius - 1) + 1):
                    temp_dissim[:, :, :, :, :, x, y, z] = batch - padded_batch[:, :, x:self.patch_size + x,
                                                                  y:self.patch_size + y, z:self.patch_size + z]

        temp_dissim = torch.exp(-1 * torch.square(temp_dissim) / self.sigmaI ** 2)
        temp_dist = self.dist
        for x in range(1 - self.radius, self.radius):
            for y in range(1 - self.radius, self.radius):
                for z in range(1 - self.radius, self.radius):
                    if x ** 2 + y ** 2 + z ** 2 < self.radius ** 2:
                        temp_dist[x + self.radius - 1, y + self.radius - 1, z + self.radius - 1] = np.exp(
                            -(x ** 2 + y ** 2 + z ** 2) / self.sigmaX ** 2)

        weight = torch.multiply(temp_dissim, temp_dist)
        del temp_dissim, temp_dist
        sum_weight = weight.sum(-1).sum(-1).sum(-1)
        return weight, sum_weight

    def forward(self, batch, preds):
        """
        Inputs:
        patch : ip patch (B x C x D x H x W)
        preds : class predictions (B x K x D x H x W)
        Output :
        soft_n_cut_loss tensor for a batch of ip patch and K-class predictions
        """
        padded_preds = self.pad(preds)
        # According to the weight formula, when Euclidean distance < r,
        # the weight is 0, so reduce the dissim matrix size to radius-1 to save time and space.
        padded_batch = self.pad(batch)
        weight, sum_weight = self._cal_weights(batch=batch, padded_batch=padded_batch)

        # too many values to unpack
        temp_cropped_seg = self.cropped_seg[:preds.shape[0], :, :, :, :, :, :, :]
        for x in range((self.radius - 1) * 2 + 1):
            for y in range((self.radius - 1) * 2 + 1):
                for z in range((self.radius - 1) * 2 + 1):
                    temp_cropped_seg[:, :, :, :, :, x, y, z] = padded_preds[:, :, x:x + preds.size()[2], y:y + preds.size()[3],
                                       z:z + preds.size()[4]]
        # cropped_seg = []
        # for x in torch.arange((self.radius - 1) * 2 + 1, dtype=torch.long):
        #     width = []
        #     for y in torch.arange((self.radius - 1) * 2 + 1, dtype=torch.long):
        #         depth = []
        #         for z in torch.arange((self.radius - 1) * 2 + 1, dtype=torch.long):
        #             depth.append(
        #                 padded_preds[:, :, x:x + preds.size()[2], y:y + preds.size()[3], z:z + preds.size()[4]])
        #         width.append(torch.stack(depth, 5))
        #     cropped_seg.append(torch.stack(width, 5))
        # cropped_seg = torch.stack(cropped_seg, 5)

        numerator = temp_cropped_seg.mul(weight)
        del temp_cropped_seg
        numerator = numerator.sum(-1).sum(-1).sum(-1).mul(preds)
        denominator = sum_weight.mul(preds)

        assocA = numerator.view(numerator.shape[0], numerator.shape[1], -1).sum(-1)
        assocV = denominator.view(denominator.shape[0], denominator.shape[1], -1).sum(-1)
        assoc = assocA.div(assocV).sum(-1)

        soft_n_cut_loss = torch.add(-assoc, self.num_classes)
        return soft_n_cut_loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, y_pred, y_true):
        return 1 - self.ssim(y_pred, y_true)
