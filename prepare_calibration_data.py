import os
import glob
import shutil
import random


TARGET_DIR = "calibration_data"

NUM_PAIRS_TO_COPY = 100

SINTEL_CLEAN_DIR = os.path.join('datasets', 'Sintel', 'training', 'clean')
SINTEL_FINAL_DIR = os.path.join('datasets', 'Sintel', 'training', 'final')
KITTI_TRAIN_DIR = os.path.join('datasets', 'KITTI', 'training', 'image_2')

SOURCE_DATASETS = [
    SINTEL_CLEAN_DIR,
    SINTEL_FINAL_DIR
]


def find_sintel_pairs(base_dir):
    """
    根据 MpiSintel 逻辑查找 Sintel 图像对
    """
    pairs = []
    if not os.path.exists(base_dir):
        print(f"警告：未找到 Sintel 路径: {base_dir}")
        return []

    scenes = os.listdir(base_dir)
    for scene in scenes:
        scene_path = os.path.join(base_dir, scene)
        if not os.path.isdir(scene_path):
            continue

        images = sorted(glob.glob(os.path.join(scene_path, '*.png')))
        for i in range(len(images) - 1):
            # 添加连续的帧
            pairs.append((images[i], images[i + 1]))

    print(f"在 {base_dir} 中找到 {len(pairs)} 个图像对。")
    return pairs


def find_kitti_pairs(base_dir):
    """
    根据 KITTI 逻辑查找 KITTI 图像对
    """
    pairs = []
    if not os.path.exists(base_dir):
        print(f"警告：未找到 KITTI 路径: {base_dir}")
        return []

    images1 = sorted(glob.glob(os.path.join(base_dir, '*_10.png')))
    images2 = sorted(glob.glob(os.path.join(base_dir, '*_11.png')))

    for img1, img2 in zip(images1, images2):
        pairs.append((img1, img2))

    print(f"在 {base_dir} 中找到 {len(pairs)} 个图像对。")
    return pairs


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"准备在 '{TARGET_DIR}' 文件夹中创建校准数据...")

    all_source_pairs = []

    # 添加 Sintel 图像对
    for sintel_dir in SOURCE_DATASETS:
        all_source_pairs.extend(find_sintel_pairs(sintel_dir))

    # 添加 KITTI 图像对
    all_source_pairs.extend(find_kitti_pairs(KITTI_TRAIN_DIR))

    if not all_source_pairs:
        print("\n错误：未在任何数据集中找到图像！")
        print(f"请确保你的数据集位于以下路径：")
        print(f"- {SINTEL_CLEAN_DIR}")
        print(f"- {SINTEL_FINAL_DIR}")
        print(f"- {KITTI_TRAIN_DIR}")
        return

    print(f"\n总共找到了 {len(all_source_pairs)} 个图像对。")
    print(f"正在随机抽取 {NUM_PAIRS_TO_COPY} 对 (共 {NUM_PAIRS_TO_COPY * 2} 个文件)...")

    random.shuffle(all_source_pairs)
    pairs_to_copy = all_source_pairs[:NUM_PAIRS_TO_COPY]

    copied_count = 0
    for img1_path, img2_path in pairs_to_copy:
        try:
            # 复制 img1
            dest1 = os.path.join(TARGET_DIR, os.path.basename(img1_path))
            if not os.path.exists(dest1):  # 避免重复复制
                shutil.copy(img1_path, dest1)
                copied_count += 1

            # 复制 img2
            dest2 = os.path.join(TARGET_DIR, os.path.basename(img2_path))
            if not os.path.exists(dest2):  # 避免重复复制
                shutil.copy(img2_path, dest2)
                copied_count += 1

        except Exception as e:
            print(f"复制文件时出错: {e}")

    print("\n---------------------------------")
    print("校准数据准备完毕")
    print(f"总共复制了 {copied_count} 个文件到 '{TARGET_DIR}' 文件夹。")
    print("---------------------------------")


if __name__ == "__main__":
    main()