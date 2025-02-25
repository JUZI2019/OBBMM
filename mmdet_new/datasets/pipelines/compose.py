# Copyright (c) OpenMMLab. All rights reserved.
import collections

from mmcv_new.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        # # 把两个分支的图像画出来看看
        # import numpy as np
        # import cv2
        # img = data['img'].data.detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # img_bg = data['img_bg'].data.detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # cv2.imwrite('./img_T.jpg', img)
        # cv2.imwrite('./img_bg_T.jpg', img_bg)
        



        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            str_ = t.__repr__()
            if 'Compose(' in str_:
                str_ = str_.replace('\n', '\n    ')
            format_string += '\n'
            format_string += f'    {str_}'
        format_string += '\n)'
        return format_string
