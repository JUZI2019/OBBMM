import torch
import time
from mmrotate.models import build_detector
from mmcv_new import Config
from mmcv_new.runner import load_checkpoint
from thop import profile

# 加载配置文件和模型
config_path = '/workstation/fyy/mm_runs/oriented_reppoints_r50_fpn_1x_sen1ship_le135/oriented_reppoints_r50_fpn_1x_sen1ship_le135.py'  # 替换为你的配置文件路径
checkpoint_path = '/workstation/fyy/mm_runs/oriented_reppoints_r50_fpn_1x_sen1ship_le135/best.pth'  # 替换为你的权重文件路径

cfg = Config.fromfile(config_path)
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
model.eval()

# 加载权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_checkpoint(model, checkpoint_path, map_location=device)
model = model.to(device)

# 输入张量
input_tensor = torch.randn(1, 3, 608, 608).to(device)  # 替换为模型的输入尺寸

# 假的 img_metas
fake_img_metas = [{
    'img_shape': (608, 608, 3),
    'ori_shape': (608, 608, 3),
    'scale_factor': 1.0
}]

# 包装 forward 函数，提供 img_metas
def forward_wrapper(model, input_tensor):
    return model.forward_dummy(input_tensor)

# 使用 forward_dummy 计算 FLOPs 和参数量
from mmrotate.models.detectors.base import BaseDetector
if isinstance(model, BaseDetector):
    model.forward = model.forward_dummy

# 计算 FLOPs 和参数量
flops, params = profile(model, inputs=(input_tensor,))
print(f"Model FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Model Parameters: {params / 1e6:.2f} M")

# 测试每百次推理时间
with torch.no_grad():
    # 预热 GPU
    for _ in range(10):
        _ = model(input_tensor)

    # 正式计时
    start_time = time.time()
    for _ in range(100):
        _ = model(input_tensor)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / 100
    print(f"Average Inference Time: {avg_inference_time * 1000:.2f} ms")
