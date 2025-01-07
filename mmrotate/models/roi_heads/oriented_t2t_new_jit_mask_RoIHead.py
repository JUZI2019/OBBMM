import torch
import torch.nn as nn
import time
import numpy as np
# from numba import jit
from mmcv_new.runner import ModuleList
# 注意：替换导入，使用 mmrotate 中的函数和装饰器
from mmrotate.core import (rbbox2result, rbbox2roi, build_assigner,
                           build_sampler)

# from mmdet_new.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
#                         build_sampler, merge_aug_bboxes, merge_aug_masks,
#                         multiclass_nms)
from ..builder import ROTATED_HEADS, build_shared_head

# from mmdet_new.models.roi_heads.base_roi_head import BaseRoIHead

# from .test_mixins import RBoxTestMixin, MaskTestMixin
# from .rotate_standard_roi_head import RotatedStandardRoIHead
from .oriented_standard_roi_head import OrientedStandardRoIHead

######### t2t model ##########
from .t2t_models.t2t_vit import T2T_module

####### evit topk model ######
import math
from functools import partial
from .evit.helpers import complement_idx
##############################

######### t2t model ##########
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., keep_rate=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

    def forward(self, x, keep_rate=None, tokens=None):
        if keep_rate is None:
            keep_rate = self.keep_rate
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        left_tokens = N - 1
        if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:  # double check the keep rate
            left_tokens = math.ceil(keep_rate * (N - 1))
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            # cls_idx = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
            # index = torch.cat([cls_idx, idx + 1], dim=1)
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            return x, index, idx, cls_attn, left_tokens

        return  x, None, None, None, left_tokens

## no topK , only reture atten score
class Token_Pair_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, tokens=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn[:,:,0,1] = 0  # cls see box as 0
        attn[:,:,1,0] = 0  # box see cls as 0
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        ##### for mask #####
        # with torch.no_grad():
        # attn[:,:,0,1] = 0  # cls not see box
        # attn[:,:,1,0] = 0  # box not see cls
        # attn[:,:,0,1].detach()
        # attn[:,:,1,0].detach()
        ####################


        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        cls_attn = attn[:, :, 0, 2:]  # [B, H, N-1]
        cls_attn = cls_attn.mean(dim=1)  # [B, N-1]

        box_attn = attn[:, :, 1, 2:]
        box_attn = box_attn.mean(dim=1)

        return  x, cls_attn, box_attn

## pair the class and bbox token
## return cls + cls_g token and bbox + bbox_g token
# @jit()
def gpu_pair(token_compare, index, B, N):
    # id = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x

    idx = np.zeros([B,N-1],dtype=np.int64)

    for b in range(B):
      box_token_cnt = 0
      cls_token_cnt = 15
      for j in index[b]:
          if token_compare[b,j]:
              idx[b,box_token_cnt] = int(j)
              box_token_cnt += 1
          else:
              idx[b,cls_token_cnt] = int(j)
              cls_token_cnt -= 1
    return idx

class Token_pair_block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Token_Pair_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_hidden_dim = mlp_hidden_dim

    def forward(self, general_token , cls_token , box_token, keep_rate=None, tokens=None, get_idx=False):

        # cls_fuse_general = torch.cat((cls_token, general_token), dim=1)
        # box_fuse_general = torch.cat((box_token, general_token), dim=1)

        cat_token = torch.cat((box_token, general_token), dim=1)
        cat_token = torch.cat((cls_token, cat_token), dim=1)


        B, N, C = cat_token.shape
        # print(N)

        # cls_tmp, cls_attn = self.attn(self.norm1(cls_fuse_general))
        # box_tmp, box_attn = self.attn(self.norm1(box_fuse_general))  # [B , N-1] N:total token nun , N-1=16

        cat_token_temp, cls_attn, box_attn = self.attn(self.norm1(cat_token))

        cat_token = cat_token + self.drop_path(cat_token_temp)

        # cls_feat = cls_fuse_general + self.drop_path(cls_tmp)  # [B, 17, 128]
        # box_feat = box_fuse_general + self.drop_path(box_tmp)

        # cls_token = cls_feat[:,0:1]
        # cls_feat = cls_feat[:,1:]

        # box_token = box_feat[:,0:1]
        # box_feat = box_feat[:,1:]


        general_token = cat_token[:,2:]
        cls_token = cat_token[:,0:1]
        box_token = cat_token[:,1:2]



        # print(cls_attn)
        sum_attn = cls_attn + box_attn
        # print(sum_attn)

        _ , index = torch.sort(sum_attn, descending=True)

        # print(index.size()) # index of largest to smallest

        # fuse_task_token = torch.empty([B,task_token_num*2,C]).to(torch.device('cuda:0'))
        # box_task_token = torch.empty([B,task_token_num,C]).to(torch.device('cuda:0'))
        # cls_task_token = torch.empty([B,task_token_num,C]).to(torch.device('cuda:0'))


        # print(cls_feat)
        #### pair the toke ALG
        # start = time.time()
        token_compare = torch.gt(box_attn,cls_attn) # True if (box_attn>cls_attn)

        index = index.cpu().numpy()
        token_compare = token_compare.cpu().numpy()



        # start = time.time()
        # idx = np.zeros([B,N-1],dtype=np.int64)
        # for b in range(B) :
        #     box_token_cnt = 0
        #     cls_token_cnt = 15
        #     for i in index[b] :


        #         if token_compare[b,i]: #> cls_attn[b,i]:

        #             idx[b,box_token_cnt] = int(i)
        #             box_token_cnt += 1
        #         else:
        #             idx[b,cls_token_cnt] = int(i)
        #             cls_token_cnt -= 1


        idx = gpu_pair(token_compare,index,B,N)
        # end = time.time()
        # print('time:',end-start)
        idx = torch.from_numpy(idx).to(torch.device('cuda'))
        idx = idx.unsqueeze(-1).expand(-1, -1, C)

        cat_token = torch.gather(general_token,dim=1,index=idx)

        # print(cls_token.size())

        cls_task_token = torch.cat((cls_token,cat_token[:,0:8,:]),dim=1)
        box_task_token = torch.cat((box_token,cat_token[:,8:16,:]),dim=1)


        return cls_task_token , box_task_token

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_rate=0.,
                 fuse_token=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop, keep_rate=keep_rate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.keep_rate = keep_rate
        self.mlp_hidden_dim = mlp_hidden_dim
        self.fuse_token = fuse_token

    def forward(self, x, keep_rate=None, tokens=None, get_idx=False):
        if keep_rate is None:
            keep_rate = self.keep_rate  # this is for inference, use the default keep rate
        B, N, C = x.shape

        tmp, index, idx, cls_attn, left_tokens = self.attn(self.norm1(x), keep_rate, tokens)
        x = x + self.drop_path(tmp)

        if index is not None:
            # B, N, C = x.shape
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            if self.fuse_token:
                compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
                non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]

                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_others], dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 1
        if get_idx and index is not None:
            return x, n_tokens, idx
        return x, n_tokens, None
##################################

@ROTATED_HEADS.register_module()
class Oriented_t2t_new_jit_mask_RoIHead(OrientedStandardRoIHead):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 version='oc'):

        super(Oriented_t2t_new_jit_mask_RoIHead, self).__init__(

            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.version = version
        ####### t2t_module #######
        # self.token_to_token = T2T_module(
        #         img_size=7, tokens_type='performer', in_chans=256, embed_dim=128, token_dim=100)
        self.keep_rate = 0.5
        # t2t_token should be checked by feature map dim
        t2t_token = 8
        # evit_token = math.ceil(self.keep_rate*t2t_token) + 2
        self.in_chans = 256
        self.token_dim = 100
        self.embed_dim = 128
        self.bbox_token =  nn.Parameter(torch.zeros(1, self.in_chans * 3 * 3)).to(torch.device('cuda'))
        self.cls_token = nn.Parameter(torch.zeros(1, self.in_chans * 3 * 3)).to(torch.device('cuda'))
        self.token_to_token = ModuleList([
                                T2T_module(
                                    img_size=7, 
                                    tokens_type='transformer', 
                                    in_chans=self.in_chans, 
                                    embed_dim=self.embed_dim, 
                                    token_dim=self.token_dim, 
                                    mask=True)])
        self.t2t_bbox_head = ModuleList([nn.Linear(9*128, 5)])
        self.token_pair = ModuleList([Token_pair_block(dim=self.embed_dim, num_heads=t2t_token)])
        ###### sen1ship dataset #####
        self.t2t_cls_head = ModuleList([nn.Linear(9*128, 2) ])


        # ori
        # self.bbox_token = [nn.Parameter(torch.zeros(1, self.in_chans * 3 * 3)).to(torch.device('cuda')) for _ in range(self.num_stages)]
        # self.cls_token = [nn.Parameter(torch.zeros(1, self.in_chans * 3 * 3)).to(torch.device('cuda')) for _ in range(self.num_stages)]
        # self.token_to_token = ModuleList([T2T_module(
        #                         img_size=7, tokens_type='transformer', in_chans=self.in_chans, embed_dim=self.embed_dim, token_dim=self.token_dim, mask=True) for _ in range(self.num_stages)])
        # # self.t2t_bbox_head = ModuleList([nn.Linear(9*128, 4) for _ in range(self.num_stages)])
        # self.t2t_bbox_head = ModuleList([nn.Linear(9*128, 5) for _ in range(self.num_stages)])

        # # norm_layer = partial(nn.LayerNorm, eps=1e-6)


        ## do the bipartite token pair ##
        # self.token_pair = ModuleList([Token_pair_block(dim=self.embed_dim, num_heads=t2t_token)  for _ in range(self.num_stages)])

        # # self.blk = Block(keep_rate=0.7, fuse_token=True)
        # # self.cls_blk = ModuleList([Block(dim=self.embed_dim, num_heads=t2t_token, keep_rate=self.keep_rate, fuse_token=True) for _ in range(self.num_stages)])
        # # self.bbox_blk = ModuleList([Block(dim=self.embed_dim, num_heads=t2t_token, keep_rate=self.keep_rate, fuse_token=True) for _ in range(self.num_stages)])
        # # self.norm = norm_layer(self.embed_dim)

        ###### aitod dataset #####
        # self.t2t_cls_head = ModuleList([nn.Linear(9*128, 9) for _ in range(self.num_stages)])
        ##### visdrone dataset ###
        # self.t2t_cls_head = ModuleList([nn.Linear(128, 11) for _ in range(self.num_stages)])
        ##########################

        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        self.init_assigner_sampler()

        self.with_bbox = True if bbox_head is not None else False
        self.with_shared_head = True if shared_head is not None else False

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)


    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # bbox_roi_extractor = self.bbox_roi_extractor[stage]
        # bbox_head = self.bbox_head[stage]
        # bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        # do not support caffe_c4 model anymore
        # cls_score, bbox_pred = bbox_head(bbox_feats)

        ####### t2t_module #######
        token_to_token = self.token_to_token
        # bbox_blk = self.bbox_blk[stage]
        t2t_bbox_head = self.t2t_bbox_head

        # cls_blk = self.cls_blk[stage]
        t2t_cls_head = self.t2t_cls_head

        cls_token = self.cls_token.expand(bbox_feats.shape[0], -1, -1)
        bbox_token = self.bbox_token.expand(bbox_feats.shape[0], -1, -1)
        token_pair = self.token_pair

        t2t_feats, cls_token, bbox_token = token_to_token(bbox_feats, cls_token=cls_token, bbox_token=bbox_token, random_shuffle_forward=True)

        ######### token pairing #############
        t2t_feats_cls, t2t_feats_bbox = token_pair(general_token=t2t_feats, cls_token=cls_token, box_token=bbox_token)

        # print('cls_token:',t2t_feats_cls.size())
        ########## evit topk ##########
        # t2t_feats_cls = torch.cat((cls_token, t2t_feats), dim=1)
        # t2t_feats_cls, cls_n_token, cls_token_idx = cls_blk(t2t_feats_cls)
        ################################
        # t2t_feats_cls = self.norm(t2t_feats_cls)
        # t2t_feats_cls = t2t_feats_cls[:, 0]
        t2t_feats_cls = torch.flatten(t2t_feats_cls, start_dim=1)
        cls_score = t2t_cls_head(t2t_feats_cls)

        ########## evit topk ##########
        # t2t_feats_bbox = torch.cat((bbox_token, t2t_feats), dim=1)
        # t2t_feats_bbox, bbox_n_token, bbox_token_idx = bbox_blk(t2t_feats_bbox)
        ################################
        # t2t_feats_bbox = self.norm(t2t_feats_bbox)
        # t2t_feats_bbox = t2t_feats_bbox[:, 0]
        t2t_feats_bbox = torch.flatten(t2t_feats_bbox, start_dim=1)
        bbox_pred = t2t_bbox_head(t2t_feats_bbox)

        # print("cls_score:", cls_score.size())
        # bbox_pred = self.t2t_bbox_head(t2t_feats)
        ##########################

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, 
                            x, 
                            sampling_results, 
                            gt_bboxes,
                            gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            TO OBB:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.


            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses
        # losses = dict()
        # for i in range(self.num_stages):
        #     self.current_stage = i
        #     rcnn_train_cfg = self.train_cfg[i]
        #     lw = self.stage_loss_weights[i]

        #     # assign gts and sample proposals
        #     sampling_results = []
        #     if self.with_bbox or self.with_mask:
        #         bbox_assigner = self.bbox_assigner[i]
        #         bbox_sampler = self.bbox_sampler[i]
        #         num_imgs = len(img_metas)
        #         if gt_bboxes_ignore is None:
        #             gt_bboxes_ignore = [None for _ in range(num_imgs)]

        #         for j in range(num_imgs):
        #             assign_result = bbox_assigner.assign(
        #                 proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
        #                 gt_labels[j])
        #             sampling_result = bbox_sampler.sample(
        #                 assign_result,
        #                 proposal_list[j],
        #                 gt_bboxes[j],
        #                 gt_labels[j],
        #                 feats=[lvl_feat[j][None] for lvl_feat in x])
        #             sampling_results.append(sampling_result)

        #     # bbox head forward and loss
        #     bbox_results = self._bbox_forward_train(i, x, sampling_results,
        #                                             gt_bboxes, gt_labels,
        #                                             rcnn_train_cfg)

        #     for name, value in bbox_results['loss_bbox'].items():
        #         losses[f's{i}.{name}'] = (
        #             value * lw if 'loss' in name else value)

        #     # mask head forward and loss
        #     if self.with_mask:
        #         mask_results = self._mask_forward_train(
        #             i, x, sampling_results, gt_masks, rcnn_train_cfg,
        #             bbox_results['bbox_feats'])
        #         for name, value in mask_results['loss_mask'].items():
        #             losses[f's{i}.{name}'] = (
        #                 value * lw if 'loss' in name else value)

        #     # refine bboxes
        #     if i < self.num_stages - 1:
        #         pos_is_gts = [res.pos_is_gt for res in sampling_results]
        #         # bbox_targets is a tuple
        #         roi_labels = bbox_results['bbox_targets'][0]
        #         with torch.no_grad():
        #             roi_labels = torch.where(
        #                 roi_labels == self.bbox_head[i].num_classes,
        #                 bbox_results['cls_score'][:, :-1].argmax(1),
        #                 roi_labels)
        #             proposal_list = self.bbox_head[i].refine_bboxes(
        #                 bbox_results['rois'], roi_labels,
        #                 bbox_results['bbox_pred'], pos_is_gts, img_metas)
        # # print(losses)
        # return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                         self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results
    def simple_test_bboxes(self,
                            x,
                            img_metas,
                            proposals,
                            rcnn_test_cfg,
                            rescale=False):
            """Test only det bboxes without augmentation.

            Args:
                x (tuple[Tensor]): Feature maps of all scale level.
                img_metas (list[dict]): Image meta info.
                proposals (List[Tensor]): Region proposals.
                rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
                rescale (bool): If True, return boxes in original image space.
                    Default: False.

            Returns:
                tuple[list[Tensor], list[Tensor]]: The first list contains \
                    the boxes of the corresponding image in a batch, each \
                    tensor has the shape (num_boxes, 5) and last dimension \
                    5 represent (cx, cy, w, h, a, score). Each Tensor \
                    in the second list is the labels with shape (num_boxes, ). \
                    The length of both lists should be equal to batch_size.
            """

            rois = rbbox2roi(proposals)
            bbox_results = self._bbox_forward(x, rois)
            img_shapes = tuple(meta['img_shape'] for meta in img_metas)
            scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposals)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)

            # some detector with_reg is False, bbox_pred will be None
            if bbox_pred is not None:
                # the bbox prediction of some detectors like SABL is not Tensor
                if isinstance(bbox_pred, torch.Tensor):
                    bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
                else:
                    bbox_pred = self.bbox_head.bbox_pred_split(
                        bbox_pred, num_proposals_per_img)
            else:
                bbox_pred = (None, ) * len(proposals)

            # apply bbox post-processing to each image individually
            det_bboxes = []
            det_labels = []
            for i in range(len(proposals)):
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
            return det_bboxes, det_labels
