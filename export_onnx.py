import torch
import torch.onnx
import os
import numpy as np
from NeuFlow.neuflow import NeuFlow



class NeuFlowONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.iters_s16 = 4
        self.iters_s8 = 7

    def forward(self, img1, img2):
        # 4. 将 init_bhwd 移入 forward
        B, _, H, W = img1.shape
        device = img1.device

        # 自动检测输入类型 (float32 or float16)
        # 当我们使用 float32 输入时, amp 会被设为 False
        is_half = (img1.dtype == torch.half)

        # 传递正确的 amp 标志给所有 init_bhwd 调用
        self.model.init_bhwd(B, H, W, device, amp=is_half)

        flow_preds_list = self.model(img1, img2,
                                     iters_s16=self.iters_s16,
                                     iters_s8=self.iters_s8)

        final_flow = flow_preds_list[-1]
        return final_flow

def main():
    # 路径配置
    CHECKPOINT_PATH = "checkpoint/step_003000.pth"
    EXPORT_PATH = "neuflow_model.onnx"  # 导出的 ONNX 文件名

    print("正在加载原始 NeuFlow 模型...")
    orig_model = NeuFlow()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    orig_model.load_state_dict(checkpoint['model'], strict=True)
    orig_model.eval()

    print("正在创建 ONNX 包装器...")
    model_to_export = NeuFlowONNXWrapper(orig_model)
    model_to_export.eval()

    # --- 随机输入 ---
    # 移除 .half()，使用默认的 float32
    # 解决 "avg_pool2d" not implemented for 'Half'
    dummy_img1 = torch.randn(1, 3, 384, 512)
    dummy_img2 = torch.randn(1, 3, 384, 512)

    dummy_inputs = (dummy_img1, dummy_img2)

    input_names = ["img1", "img2"]
    output_names = ["final_flow"]

    print(f"开始导出 ONNX 模型到 {EXPORT_PATH} ... (使用 float32)")

    torch.onnx.export(
        model_to_export,
        dummy_inputs,
        EXPORT_PATH,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,

        dynamic_axes={
            "img1": {0: 'batch_size', 2: 'height', 3: 'width'},
            "img2": {0: 'batch_size', 2: 'height', 3: 'width'},
            "final_flow": {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    print(f"\n导出成功！模型已保存到 {EXPORT_PATH}")
    print("可以使用 Netron 打开此文件来可视化模型结构。")


if __name__ == "__main__":
    main()