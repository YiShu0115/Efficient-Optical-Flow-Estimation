import torch
import onnxruntime as ort
import numpy as np
import os
from data_utils import datasets
from data_utils import frame_utils

from data_utils import flow_viz

def write_flow_flo(filename, flow):
    # flow shape 是 (H, W, 2)
    if flow.shape[-1] != 2:
        raise ValueError("Flow data must have shape (H, W, 2)")

    flow = flow.astype(np.float32)
    h, w = flow.shape[:2]

    with open(filename, 'wb') as f:
        # 写入 magic number
        np.array([202021.25], dtype=np.float32).tofile(f)
        # 写入尺寸
        np.array([w, h], dtype=np.int32).tofile(f)
        # 写入数据
        flow.tofile(f)

ONNX_PATH = "neuflow_model.onnx"
session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name1 = session.get_inputs()[0].name
input_name2 = session.get_inputs()[1].name
output_name = session.get_outputs()[0].name

# --- 加载 Sintel 测试集 ---
# 处理 clean 和 final 两种
sintel_dstype = 'clean'  # 或 'final'
output_path = f"./sintel_submission/{sintel_dstype}"
os.makedirs(output_path, exist_ok=True)

test_dataset = datasets.MpiSintel(split='test', dstype=sintel_dstype)
print(f"Sintel ({sintel_dstype}) 测试集样本数: {len(test_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for test_id in range(len(test_dataset)):
    # 加载数据 (img1, img2 已经是 0-255 范围的 float32 Tensor)
    # extra_info 包含 (scene, frame_id)
    img1, img2, extra_info = test_dataset[test_id]
    scene, frame_id = extra_info

    img1 = img1[None].to(device)
    img2 = img2[None].to(device)

    padding_factor = 16  #
    padder = frame_utils.InputPadder(img1.shape, mode='sintel', padding_factor=padding_factor)
    img1_pad, img2_pad = padder.pad(img1, img2)

    # ONNX 推理
    img1_np = img1_pad.cpu().numpy()
    img2_np = img2_pad.cpu().numpy()

    results = session.run(
        [output_name],
        {input_name1: img1_np, input_name2: img2_np}
    )

    flow_pr = torch.from_numpy(results[0]).to(device)
    flow = padder.unpad(flow_pr)

    flow_numpy = flow[0].permute(1, 2, 0).cpu().numpy()  # [B,C,H,W] -> [H,W,C]

    scene_path = os.path.join(output_path, scene)
    os.makedirs(scene_path, exist_ok=True)

    # Sintel 帧名格式 "frame_0001.flo"
    output_filename = os.path.join(scene_path, f"frame_{frame_id + 1:04d}.flo")

    write_flow_flo(output_filename, flow_numpy)
    output_vis_filename = output_filename.replace('.flo', '.png')
    flow_viz.save_vis_flow_tofile(flow_numpy, output_vis_filename)

    print(f"已处理并保存 (模拟): {output_filename}")

print("Sintel 测试集处理完毕。")