# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .dual_branch_mmroate_base import Dual_Branch_RotatedBaseDetector
from mmrotate.models.utils.conv import DWConv, Conv
from mmcv_new.cnn import ConvModule

import torch.nn as nn
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

@ROTATED_DETECTORS.register_module()
class Dual_Branch_RotatedTwoStageDetector(Dual_Branch_RotatedBaseDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(Dual_Branch_RotatedTwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # self.cv1 = DWConv(512, 256, 1, 1)
        self.cv1 = Conv(512, 256, 1, 1)
        # self.cv1 =  nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1),
        #                 nn.BatchNorm2d(256),
        #                 nn.ReLU(inplace=True))

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None
    

    def extract_feat(self, img):
        # 原来的 
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x


    def extract_concat_feat(self, img, img_bg):
        # TODO:两个分支特征融合，目前这里先这么写
        # 原来的 
        # """Directly extract features from the backbone+neck."""
        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
        # return x


        # 在FPN后concat，精度不行
        x = self.backbone(img)
        x_bg = self.backbone(img_bg)
        # # 把两个分支的图像画出来看看
        # for idx in range(x[0].shape[1]):
        #     temp = x[0][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-x1P1.png'.format(idx)), temp)
        # # 把两个分支的图像画出来看看
        # for idx in range(x_bg[0].shape[1]):
        #     temp = x_bg[0][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-x2P1.png'.format(idx)), temp)
        # # 把两个分支的图像画出来看看
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-x1P2.png'.format(idx)), temp)
        # # 把两个分支的图像画出来看看
        # for idx in range(x_bg[1].shape[1]):
        #     temp = x_bg[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-x2P2.png'.format(idx)), temp)
        if self.with_neck:
            x = self.neck(x)
            x_bg = self.neck(x_bg)

        # i = img.data.cpu().numpy()
        # i_bg = img_bg.cpu().numpy()
        # plt.imsave('./img.jpg', i[0][0,:,:])
        # plt.imsave('./img_bg.jpg', i_bg[0][0,:,:])

        # # 把两个分支的图像画出来看看
        # for idx in range(x[0].shape[1]):
        #     temp = x[0][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-x1P1.png'.format(idx)), temp)
        # # 把两个分支的图像画出来看看
        # for idx in range(x_bg[0].shape[1]):
        #     temp = x_bg[0][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-x2P1.png'.format(idx)), temp)
        # # 把两个分支的图像画出来看看
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-x1P2.png'.format(idx)), temp)
        # # 把两个分支的图像画出来看看
        # for idx in range(x_bg[1].shape[1]):
        #     temp = x_bg[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-x2P2.png'.format(idx)), temp)
        
        # 使用中间变量来存储结果
        x_modified = []
        for i in range(len(x)):
            x_i_modified = []
            for batch_id in range(x[i].shape[0]):
                x1 = x[i][batch_id,:,:,:].unsqueeze(0)
                x2 = x_bg[i][batch_id,:,:,:].unsqueeze(0)
                x_cat = torch.cat([x1, x2], dim=1)
                # x_modified_batch = x_cat.squeeze(0)
                x_modified_batch = self.cv1(x_cat).squeeze(0)

                x_i_modified.append(x_modified_batch)
            x_modified.append(torch.stack(x_i_modified))

        # # # 画出来看看
        # # for idx in range(x_modified[0].shape[1]):
        # #     temp = x_modified[0][0,idx,:,:]
        # #     temp = temp.data.cpu().numpy()
        # #     plt.imsave(os.path.join('dilation', '{}-combineP1.png'.format(idx)), temp)
        # # for idx in range(x_modified[1].shape[1]):
        # #     temp = x_modified[1][0,idx,:,:]
        # #     temp = temp.data.cpu().numpy()
        # #     plt.imsave(os.path.join('dilation', '{}-combineP2.png'.format(idx)), temp)
        return x_modified
        
    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(imgs[i]) for i in len(imgs)]
    
    
    def extract_concat_feats(self, imgs, imgs_bg):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        assert len(imgs)==len(imgs_bg)  # 确保两个分支图片数量一致
        return [self.extract_concat_feat(imgs[i], imgs_bg[i]) for i in len(imgs)]
    
    def forward_dummy(self, img, img_bg):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        x_bg = self.extract_feat(img_bg)
        x_cat = [self.cv1(torch.cat([x[i], x_bg[i]], dim=1)) for i in range(len(x))]
        # x_cat = [ x[i]+ x_bg[i] for i in range(len(x))]
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 5).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x_cat, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_bg,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # 单分支特征
        x = self.extract_feat(img)
        x_bg = self.extract_feat(img_bg)
        x_cat = [self.cv1(torch.cat([x[i], x_bg[i]], dim=1)) for i in range(len(x))]
        # x_cat = [ x[i]+ x_bg[i] for i in range(len(x))]
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                # x_cat,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x_cat, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_bg,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        x_bg = self.extract_feat(img_bg)
        x_cat = [self.cv1(torch.cat([x[i], x_bg[i]], dim=1)) for i in range(len(x))]
        # x_cat = [ x[i]+ x_bg[i] for i in range(len(x))]
        

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x_cat, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_bg, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        x_bg = self.extract_feat(img_bg)
        x_cat = [self.cv1(torch.cat([x[i], x_bg[i]], dim=1)) for i in range(len(x))]
        # x_cat = [ x[i]+ x_bg[i] for i in range(len(x))]

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x_cat, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, imgs_bg, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        x_bg = self.extract_feats(imgs_bg)
        x_cat = [[self.cv1(torch.cat([x[n][i], x_bg[n][i]], dim=1)) for i in range(len(x))] for n in range(len(imgs)) ]
        # x_cat = [[x[n][i]+ x_bg[n][i] for i in range(len(x))] for n in range(len(imgs)) ]

        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x_cat, proposal_list, img_metas, rescale=rescale)
