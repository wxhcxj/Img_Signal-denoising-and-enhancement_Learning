import argparse
from pathlib import Path

import cv2
import numpy as np


def kalman_filter_1d(z: np.ndarray, q: float, r: float, x0: float | None = None, p0: float = 1.0, a: float = 1.0, h: float = 1.0) -> np.ndarray:
    """一维卡尔曼滤波，输入/输出均为归一化域[0,1]。"""
    n = z.shape[0]
    x_hat = np.zeros(n, dtype=np.float32)

    x_prev = float(z[0] if x0 is None else x0)
    p_prev = float(p0)

    for k in range(n):
        # Predict
        x_pred = a * x_prev
        p_pred = a * p_prev * a + q

        # Update
        y = z[k] - h * x_pred
        s = h * p_pred * h + r
        k_gain = (p_pred * h) / s

        x_new = x_pred + k_gain * y
        p_new = (1.0 - k_gain * h) * p_pred

        x_hat[k] = x_new
        x_prev, p_prev = x_new, p_new

    return x_hat


def kalman_denoise_gray(image: np.ndarray, mode: str, q: float, r: float, p0: float = 1.0, a: float = 1.0, h: float = 1.0) -> np.ndarray:
    """灰度图卡尔曼去噪。

    mode:
    - row: 逐行
    - col: 逐列
    - raster: 像素序列化（按行展开）
    """
    if image.ndim != 2:
        raise ValueError("kalman_denoise_gray 仅支持灰度图(2D)")
    if image.dtype != np.uint8:
        raise ValueError("image 必须是 uint8")

    img = image.astype(np.float32) / 255.0
    hgt, wdt = img.shape
    out = np.zeros_like(img, dtype=np.float32)

    if mode == "row":
        for i in range(hgt):
            out[i, :] = kalman_filter_1d(img[i, :], q=q, r=r, x0=None, p0=p0, a=a, h=h)

    elif mode == "col":
        for j in range(wdt):
            out[:, j] = kalman_filter_1d(img[:, j], q=q, r=r, x0=None, p0=p0, a=a, h=h)

    elif mode == "raster":
        seq = img.reshape(-1)
        den_seq = kalman_filter_1d(seq, q=q, r=r, x0=None, p0=p0, a=a, h=h)
        out = den_seq.reshape(hgt, wdt)

    else:
        raise ValueError(f"不支持的 mode: {mode}")

    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).round().astype(np.uint8)


def kalman_denoise_color(image: np.ndarray, mode: str, q: float, r: float, p0: float = 1.0, a: float = 1.0, h: float = 1.0) -> np.ndarray:
    """彩色图逐通道卡尔曼去噪。"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("kalman_denoise_color 仅支持3通道图像")
    if image.dtype != np.uint8:
        raise ValueError("image 必须是 uint8")

    channels = cv2.split(image)
    den_channels = [kalman_denoise_gray(ch, mode=mode, q=q, r=r, p0=p0, a=a, h=h) for ch in channels]
    return cv2.merge(den_channels)


def main() -> None:
    parser = argparse.ArgumentParser(description="卡尔曼滤波图像去噪（支持 row/col/raster 三种建模方式）")
    parser.add_argument("--input_dir", type=str, default="data/noisy", help="待去噪图像目录")
    parser.add_argument("--output_dir", type=str, default="data/denoised/kalman", help="输出目录")
    parser.add_argument("--mode", type=str, default="row", choices=["row", "col", "raster"], help="建模方式")

    # 推荐默认参数：A=1, H=1, P0=1, Q=1e-4, R=0.01
    parser.add_argument("--a", type=float, default=1.0, help="状态转移系数 A")
    parser.add_argument("--h", type=float, default=1.0, help="观测系数 H")
    parser.add_argument("--q", type=float, default=1e-4, help="过程噪声方差 Q")
    parser.add_argument("--r", type=float, default=0.01, help="观测噪声方差 R")
    parser.add_argument("--p0", type=float, default=1.0, help="初始误差协方差 P0")

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
            den = kalman_denoise_gray(img, mode=args.mode, q=args.q, r=args.r, p0=args.p0, a=args.a, h=args.h)
        elif img.ndim == 3 and img.shape[2] == 3:
            den = kalman_denoise_color(img, mode=args.mode, q=args.q, r=args.r, p0=args.p0, a=args.a, h=args.h)
        else:
            print(f"[跳过] 不支持通道格式: {img_path}, shape={img.shape}")
            continue

        out_name = (
            f"{img_path.stem}_kalman_{args.mode}"
            f"_a{args.a:.2f}_h{args.h:.2f}_q{args.q:.0e}_r{args.r:.3f}_p0{args.p0:.2f}{img_path.suffix.lower()}"
        )
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), den)
        print(f"[保存] {out_path}")


if __name__ == "__main__":
    main()
