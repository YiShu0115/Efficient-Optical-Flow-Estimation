import onnxruntime as ort
import numpy as np

def main():
    ONNX_PATH = "neuflow_model.onnx"
    print(f"正在加载 ONNX 模型: {ONNX_PATH}")

    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

    input_name1 = session.get_inputs()[0].name
    input_name2 = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name

    print(f"输入: {input_name1}, {input_name2}")
    print(f"输出: {output_name}")


    # 创建一个 (B=1, C=3, H=480, W=640) 的随机输入
    batch_size = 1
    height = 480
    width = 640


    img1_data = np.random.rand(batch_size, 3, height, width).astype(np.float32)
    img2_data = np.random.rand(batch_size, 3, height, width).astype(np.float32)

    print(f"准备输入数据: img1.shape={img1_data.shape}, img2.shape={img2_data.shape}")

    results = session.run(
        [output_name],
        {
            input_name1: img1_data,
            input_name2: img2_data
        }
    )

    final_flow_output = results[0]

    print("\n推理成功！")
    print(f"输出的光流 (flow) shape: {final_flow_output.shape}")
    # print(final_flow_output)


if __name__ == "__main__":
    main()