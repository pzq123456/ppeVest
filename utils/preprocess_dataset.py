import cv2
import numpy as np
import random
import os
import sys
from pathlib import Path
import argparse
from hashlib import md5

# 全局尺寸变量
TARGET_SIZE = 320


def imread_unicode(path):
    """支持中文路径的 imread"""
    # 直接用 np.fromfile + imdecode，完全绕过 cv2.imread
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if len(data) == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"读取失败 {path}: {e}")
        return None


def imwrite_unicode(path, img):
    """支持中文路径的 imwrite"""
    try:
        ext = Path(path).suffix
        if not ext:
            ext = '.jpg'
        success, encoded = cv2.imencode(ext, img)
        if success:
            encoded.tofile(str(path))
            return True
    except Exception as e:
        print(f"写入失败 {path}: {e}")
    return False


def list_images_unicode(directory):
    """列出目录下所有图像文件（支持中文路径）"""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    
    try:
        # 使用 os.scandir 遍历（更好地处理中文）
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in extensions:
                        images.append(Path(entry.path))
    except Exception as e:
        print(f"遍历目录失败 {directory}: {e}")
    
    return images


def letterbox(img, pad_color, interp=None):
    """letterbox 补边，指定颜色"""
    h, w = img.shape[:2]
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if interp is None:
        interp = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])

    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    top = (TARGET_SIZE - new_h) // 2
    bottom = TARGET_SIZE - new_h - top
    left = (TARGET_SIZE - new_w) // 2
    right = TARGET_SIZE - new_w - left

    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=pad_color)


def stretch(img, interp=None):
    """直接拉伸"""
    if interp is None:
        interp = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    return cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=interp)


def process_image(img):
    """处理单张图像，随机返回 1 个变体（4 选 1），避免数据集膨胀。

    全数据集在统计意义上覆盖 4 种 DeepStream 相关预处理模式，
    但每张图只落盘 1 个变体，膨胀系数从 4x 降为 1x。
    """
    r = random.random()
    if r < 0.25:
        # 黑色 letterbox（nvinfer 默认）
        return [letterbox(img, (0, 0, 0))], ["black"]
    if r < 0.5:
        # 灰色 letterbox（ultralytics 默认）
        return [letterbox(img, (114, 114, 114))], ["gray"]
    if r < 0.75:
        # 随机色 letterbox
        rand_color = tuple(np.random.randint(0, 256, size=3).tolist())
        return [letterbox(img, rand_color)], ["rand"]
    # 直接拉伸
    return [stretch(img)], ["stretch"]


def apply_augmentations(img):
    """随机数据增强：翻转、旋转、亮度/对比度、色调/饱和度、噪声、模糊"""
    img = img.copy()
    h, w = img.shape[:2]

    # 水平翻转
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # 小角度旋转（反射填充）
    angle = random.uniform(-15, 15)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 亮度/对比度
    alpha = random.uniform(0.7, 1.3)
    beta = random.uniform(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 色调/饱和度
    if random.random() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-15, 15)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 高斯噪声
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 10), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 轻微模糊
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def process_image_aug(img, n_variants=2):
    """增强模式：每张图生成 n_variants 个变体。

    每个变体 = 随机增强 + 4 选 1 尺寸归一化（letterbox/stretch）。
    """
    variants, names = [], []
    for i in range(n_variants):
        aug = apply_augmentations(img)
        r = random.random()
        if r < 0.25:
            variants.append(letterbox(aug, (0, 0, 0)))
        elif r < 0.5:
            variants.append(letterbox(aug, (114, 114, 114)))
        elif r < 0.75:
            rand_color = tuple(np.random.randint(0, 256, size=3).tolist())
            variants.append(letterbox(aug, rand_color))
        else:
            variants.append(stretch(aug))
        names.append(f"aug{i}")
    return variants, names


def iter_split_roots(input_path):
    """返回 [(split 标签, split 目录)]。

    有 train/val 子目录则按划分处理；否则视为扁平结构，整体作为一个 split。
    """
    splits = []
    for split in ['train', 'val']:
        split_input = input_path / split
        if split_input.exists() and any(d.is_dir() for d in split_input.iterdir()):
            splits.append((split, split_input))
    if not splits:
        splits.append((None, input_path))
    return splits


def preprocess_dataset(input_dir, output_dir, seed=42, mode='preprocess', n_variants=2):
    """遍历数据集，生成变体并保存

    mode: 'preprocess' 每张图 1 个 4 选 1 变体；'aug' 每张图 n_variants 个增强变体。
    """
    random.seed(seed)
    np.random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    total_orig = 0
    total_variants = 0
    failed = 0

    for split, split_input in iter_split_roots(input_path):
        if split is None:
            # 扁平结构：类别目录直接放在输出根目录下
            split_output = output_path
            split_label = ''
        else:
            split_output = output_path / split
            split_label = f"{split}/"
        split_output.mkdir(parents=True, exist_ok=True)

        # 遍历类别目录
        try:
            class_dirs = [d for d in split_input.iterdir() if d.is_dir()]
        except Exception as e:
            print(f"遍历 {split_input} 失败: {e}")
            continue

        for class_dir in class_dirs:
            class_output = split_output / class_dir.name
            class_output.mkdir(parents=True, exist_ok=True)

            # 使用 os.scandir 列出图像（支持中文）
            img_files = list_images_unicode(class_dir)

            print(f"{split_label}{class_dir.name}: {len(img_files)} 张")

            for idx, img_file in enumerate(img_files):
                img = imread_unicode(img_file)
                if img is None:
                    failed += 1
                    if failed <= 5:  # 只打印前5个失败
                        print(f"  跳过无法读取: {img_file.name}")
                    continue

                # 确定性命名：基于索引
                safe_name = f"{idx:06d}"

                # 生成变体
                if mode == 'aug':
                    variants, names = process_image_aug(img, n_variants)
                else:
                    variants, names = process_image(img)

                for variant, vname in zip(variants, names):
                    out_file = class_output / f"{safe_name}_{vname}.jpg"
                    imwrite_unicode(out_file, variant)

                total_orig += 1
                total_variants += len(variants)

            print(f"  成功处理: {total_orig}")

    print(f"\n原始图像: {total_orig}")
    print(f"生成变体: {total_variants}")
    if failed > 0:
        print(f"失败: {failed}")
    print(f"每张原始图像生成 {total_variants // max(1, total_orig)} 个变体")


def main():
    # 设置 stdout 编码
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='dataset/vest_cls2')
    parser.add_argument('--output', default='dataset/vest_cls2_ds')
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['preprocess', 'aug'], default='preprocess',
                        help='preprocess: 4 选 1 尺寸归一化; aug: 随机增强 + 尺寸归一化')
    parser.add_argument('--n', type=int, default=2, help='aug 模式下每张原图生成的变体数')
    args = parser.parse_args()

    global TARGET_SIZE
    TARGET_SIZE = args.size

    print(f"目标尺寸: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"随机种子: {args.seed}")
    print(f"模式: {args.mode}")
    if args.mode == 'aug':
        print(f"每张原图变体数: {args.n}")
    print(f"输入: {args.input} -> 输出: {args.output}")
    preprocess_dataset(args.input, args.output, seed=args.seed,
                       mode=args.mode, n_variants=args.n)
    print("完成")


if __name__ == "__main__":
    main()