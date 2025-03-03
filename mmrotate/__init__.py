# Copyright (c) OpenMMLab. All rights reserved.
import mmcv_new
import mmdet_new

from .core import *  # noqa: F401, F403
from .datasets import *  # noqa: F401, F403
from .models import *  # noqa: F401, F403
from .version import __version__, short_version


def digit_version(version_str):
    """Digit version."""
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '1.5.3'
mmcv_maximum_version = '1.8.0'
mmcv_version = digit_version(mmcv_new.__version__)

assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv_new.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

mmdet_minimum_version = '2.25.1'
mmdet_maximum_version = '3.0.0'
mmdet_version = digit_version(mmdet_new.__version__)

assert (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version < digit_version(mmdet_maximum_version)), \
    f'MMDetection=={mmdet_new.__version__} is used but incompatible. ' \
    f'Please install mmdet>={mmdet_minimum_version}, <{mmdet_maximum_version}.'

__all__ = ['__version__', 'short_version']
