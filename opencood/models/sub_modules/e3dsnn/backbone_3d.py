# Adapted from bollossom/E-3DSNN, commit dbe5d1731d3204850bd591a18f7255ca2d1b6712.
# Apache-2.0; see LICENSE and NOTICE in this directory.
# Changes: standalone imports, removal of unused classes/debug print,
# explicit sparse replace_feature for the same residual sum.
from functools import partial
import numpy as np
import torch
from torch import nn
import spconv.pytorch as spconv

class ReLUX(nn.Module):
    def __init__(self, thre=4):
        super(ReLUX, self).__init__()
        self.thre = thre

    def forward(self, input):
        return torch.clamp(input, 0, self.thre)

relu4 = ReLUX(thre=4)

import torch


class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens=4):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(relu4(input) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None
class Multispike(nn.Module):
    def __init__(self, lens=4, spike=multispike):
        super().__init__()
        self.lens = lens
        self.spike = spike

    def forward(self, inputs):
        return self.spike.apply(inputs)

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        Multispike(),
        conv,
        norm_fn(out_channels),
    )

    return m


class VoxelBackBone8x_3dv_snn(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.in_channels = input_channels
        self.num_stages = 4
        self.embed_channels = 16
        embed_channels=self.embed_channels
        enc_channels=[16,32,64,64]
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        enc_depth=[2, 2, 2, 2]
        groups=[4, 4, 8, 16]
        enc_num_ref=[16, 16, 16, 16]
        point_grid_size=[[8, 12, 16, 16], [6, 9, 12, 12], [4, 6, 8, 8], [3, 4, 6, 6]]
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            Multispike(),
            spconv.SubMConv3d(16, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            Multispike(),
            spconv.SubMConv3d(16, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
      
        self.conv2 = DonwBlock(
                    in_channels=embed_channels,
                    embed_channels=enc_channels[1],
                    depth=enc_depth[1],
                    norm_fn=norm_fn,
                    groups=groups[1],
                    point_grid_size=point_grid_size[1],
                    num_ref=enc_num_ref[1],
                    sp_indice_key="spconv2",
                    sub_indice_key="subm2",
                )
        self.conv3 = DonwBlock(
                    in_channels=enc_channels[1],
                    embed_channels=enc_channels[2],
                    depth=enc_depth[2],
                    norm_fn=norm_fn,
                    groups=groups[2],
                    point_grid_size=point_grid_size[2],
                    num_ref=enc_num_ref[2],
                    sp_indice_key="spconv3",
                    sub_indice_key="subm3",
                )

        self.conv4 = DonwBlock(
                    in_channels=enc_channels[2],
                    embed_channels=enc_channels[3],
                    depth=enc_depth[3],
                    norm_fn=norm_fn,
                    groups=groups[3],
                    point_grid_size=point_grid_size[3],
                    num_ref=enc_num_ref[3],
                    sp_indice_key="spconv4",
                    sub_indice_key="subm4",
                )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            Multispike(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        # Debug printing removed for training.
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        # print('1',x_conv1.features.shape)
        # print('2',x_conv2.features.shape)
        # print('3',x_conv3.features.shape)
        # print('4',x_conv4.features.shape)
        # exit(0)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict
class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_fn=None,
        indice_key=None,
        depth=4,
        groups=None,
        grid_size=None,
        bias=False,
    ):
        super().__init__()
        assert embed_channels % groups == 0
        self.groups = groups
        self.embed_channels = embed_channels
        self.proj = nn.ModuleList()
        self.grid_size = grid_size
        self.block = spconv.SparseSequential(
            Multispike(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels))
        self.voxel_block = spconv.SparseSequential(
            Multispike(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
            Multispike(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
        )

    def forward(self, x):
        feat = x
        branch = self.block(x)
        feat = branch.replace_feature(branch.features + x.features)
        res = feat
        x = feat
        x = self.voxel_block(x)
        x = x.replace_feature(x.features + res.features)
        return x


class DonwBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        depth,
        sp_indice_key,
        point_grid_size,
        num_ref=16,
        groups=None,
        norm_fn=None,
        sub_indice_key=None,
    ):
        super().__init__()
        self.num_ref = num_ref
        self.depth = depth
        self.point_grid_size = point_grid_size
        self.down = spconv.SparseSequential(
            Multispike(),
            spconv.SparseConv3d(
                in_channels,
                embed_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                indice_key=sp_indice_key,
                bias=False,
            ),
            norm_fn(embed_channels),
        )
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                BasicBlock(
                    in_channels=embed_channels,
                    embed_channels=embed_channels,
                    depth=len(point_grid_size) + 1,
                    groups=groups,
                    grid_size=point_grid_size,
                    norm_fn=norm_fn,
                    indice_key=sub_indice_key,
                )
            )

    def forward(self, x):
        x = self.down(x)
        for block in self.blocks:
            x = block(x)
        return x




