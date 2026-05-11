import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import wiener


def wiener_denoise_gray(image: np.ndarray, kernel_size: int = 5, noise_var: float | None = None) -> np.ndarray:
    """对灰度图做维纳滤波。

    参数:
    - image: uint8, 2D
    - kernel_size: 邻域窗口大小
    - noise_var: 噪声方差（在归一化域[0,1]），None 表示自动估计
    """
    if image.ndim != 2:
        raise ValueError("wiener_denoise_gray 仅接受灰度图(2D)")
    if image.dtype != np.uint8:
        raise ValueError("image 必须是 uint8")

    img = image.astype(np.float32) / 255.0

    # scipy.signal.wiener 中 noise 参数对应噪声功率（方差）
    den = wiener(img, mysize=(kernel_size, kernel_size), noise=noise_var)
    den = np.clip(den, 0.0, 1.0)
    return (den * 255.0).round().astype(np.uint8)


def wiener_denoise_color(image: np.ndarray, kernel_size: int = 5, noise_var: float | None = None) -> np.ndarray:
    """对彩色图逐通道做维纳滤波。"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("wiener_denoise_color 仅接受3通道彩色图")
    if image.dtype != np.uint8:
        raise ValueError("image 必须是 uint8")

    channels = cv2.split(image)
    denoised_channels = [wiener_denoise_gray(ch, kernel_size, noise_var) for ch in channels]
    return cv2.merge(denoised_channels)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用维纳滤波对图像去噪")
    parser.add_argument("--input_dir", type=str, default="data/noisy", help="待去噪图像目录")
    parser.add_argument("--output_dir", type=str, default="data/denoised/wiener", help="输出目录")
    parser.add_argument("--kernel_size", type=int, default=5, help="滤波窗口大小，建议奇数")
    parser.add_argument(
        "--noise_var",
        type=float,
        default=None,
        help="噪声方差(归一化域)。默认 None=自动估计，可设为0.01/0.02/0.05",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])

    if not image_paths:
        raise FileNotFoundError(f"在 {input_dir} 未找到图像")

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue

        if img.ndim == 2:
            den = wiener_denoise_gray(img, kernel_size=args.kernel_size, noise_var=args.noise_var)
        elif img.ndim == 3 and img.shape[2] == 3:
            den = wiener_denoise_color(img, kernel_size=args.kernel_size, noise_var=args.noise_var)
        else:
            print(f"[跳过] 不支持通道格式: {img_path}, shape={img.shape}")
            continue

        out_name = f"{img_path.stem}_wiener_k{args.kernel_size}{img_path.suffix.lower()}"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), den)
        print(f"[保存] {out_path}")


if __name__ == "__main__":
    main()
