import onnxruntime.quantization as ort_quant
import onnxruntime as ort
import numpy as np
import os
import glob

try:
    from data_utils import frame_utils
except ImportError:
    print("错误：确保 data_utils 文件夹在 Python 路径下")
    exit()


class NeuFlowDataReader(ort_quant.CalibrationDataReader):
    def __init__(self, calibration_image_folder, input_names):
        self.image_folder = calibration_image_folder
        self.input_names = input_names

        img1_files = sorted(glob.glob(os.path.join(calibration_image_folder, "*_img1.png")))
        img2_files = sorted(glob.glob(os.path.join(calibration_image_folder, "*_img2.png")))
        self.image_pairs = list(zip(img1_files, img2_files))

        if not self.image_pairs:
            all_files = sorted(glob.glob(os.path.join(calibration_image_folder, "*.png")))
            if len(all_files) >= 2:
                self.image_pairs = [(all_files[i], all_files[i + 1]) for i in range(0, len(all_files) - 1, 2)]

        if not self.image_pairs:
            print(f"错误：在 {calibration_image_folder} 中未找到校准图像对！")
            print("文件命名为 xxx_img1.png / xxx_img2.png 或 00001.png / 00002.png")

        print(f"找到了 {len(self.image_pairs)} 对校准图像。")
        self.data_iter = iter(self.image_pairs)

    def get_next(self):
        try:
            f1_path, f2_path = next(self.data_iter)

            img1 = frame_utils.read_gen(f1_path)  #
            img2 = frame_utils.read_gen(f2_path)  #

            img1 = np.array(img1).astype(np.uint8)[..., :3]  #
            img2 = np.array(img2).astype(np.uint8)[..., :3]  #

            img1_data = img1.transpose(2, 0, 1)  # HWC -> CHW
            img2_data = img2.transpose(2, 0, 1)  #

            img1_data = img1_data[np.newaxis, :, :, :].astype(np.float32)  #
            img2_data = img2_data[np.newaxis, :, :, :].astype(np.float32)  #

            #  ONNX 模型不包含 Padding，所以校准数据必须 Padded
            padding_factor = 16  #
            B, C, H, W = img1_data.shape

            pad_ht = (((H // padding_factor) + 1) * padding_factor - H) % padding_factor  #
            pad_wd = (((W // padding_factor) + 1) * padding_factor - W) % padding_factor  #

            # 使用 'sintel' 模式的 padding
            _pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

            # F.pad 'replicate' 对应 np.pad 'edge'
            pad_dims = ((0, 0), (0, 0), (_pad[2], _pad[3]), (_pad[0], _pad[1]))
            img1_data = np.pad(img1_data, pad_dims, mode='edge')
            img2_data = np.pad(img2_data, pad_dims, mode='edge')


            return {self.input_names[0]: img1_data, self.input_names[1]: img2_data}

        except StopIteration:
            return None

    def rewind(self):
        self.data_iter = iter(self.image_pairs)

def main():
    # --- 配置 ---
    model_fp32_path = "neuflow_model.onnx"
    model_quant_path = "neuflow_model.quant.int8.onnx"  # 量化后的模型保存路径
    calibration_data_folder = "./calibration_data"

    if not os.path.exists(calibration_data_folder) or not os.listdir(calibration_data_folder):
        print(f"错误：校准文件夹 '{calibration_data_folder}' 为空或不存在。")
        print("请准备校准数据。")
        return

    session = ort.InferenceSession(model_fp32_path, providers=['CPUExecutionProvider'])
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"模型输入: {input_names}")

    calib_reader = NeuFlowDataReader(calibration_data_folder, input_names)

    print("开始静态量化 (Static Quantization)")
    ort_quant.quantize_static(
        model_input=model_fp32_path,
        model_output=model_quant_path,
        calibration_data_reader=calib_reader,

        quant_format=ort_quant.QuantFormat.QDQ,
        activation_type=ort_quant.QuantType.QInt8,  # 激活值量化为 INT8
        weight_type=ort_quant.QuantType.QInt8,  # 权重量化为 INT8

        # 确保模型中的关键算子（如 AveragePool）被量化
        extra_options={'OpTypesToQuantize': ['Conv', 'MatMul', 'Add', 'AveragePool']}
    )

    print(f"\n量化完成！模型已保存到: {model_quant_path}")

    fp32_size = os.path.getsize(model_fp32_path) / (1024 * 1024)
    quant_size = os.path.getsize(model_quant_path) / (1024 * 1024)
    print(f"原始模型大小 (FP32): {fp32_size:.2f} MB")
    print(f"量化模型大小 (INT8): {quant_size:.2f} MB (缩小了约 {fp32_size / quant_size:.1f} 倍)")


if __name__ == "__main__":
    main()