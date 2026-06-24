import argparse
from pathlib import Path

import cv2
import numpy as np


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, var: float = 0.01, seed: int | None = None) -> np.ndarray:
    """给图像添加高斯噪声。

    参数约定：
    - 输入图像是 uint8, 范围 [0, 255]
    - mean/var 在归一化域 [0,1] 上定义
    """
    if image.dtype != np.uint8:
        raise ValueError("image 必须是 uint8 类型")

    rng = np.random.default_rng(seed)

    image_norm = image.astype(np.float32) / 255.0
    sigma = np.sqrt(var)
    noise = rng.normal(loc=mean, scale=sigma, size=image_norm.shape).astype(np.float32)

    noisy = np.clip(image_norm + noise, 0.0, 1.0)
    noisy_uint8 = (noisy * 255.0).round().astype(np.uint8)
    return noisy_uint8


def parse_variances(variances_str: str) -> list[float]:
    vals = []
    for x in variances_str.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError("variances 不能为空")
    for v in vals:
        if v < 0:
            raise ValueError(f"方差必须 >= 0，收到 {v}")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="为文件夹内图像批量添加高斯噪声")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="原图目录")
    parser.add_argument("--output_dir", type=str, default="data/noisy", help="噪声图输出目录")
    parser.add_argument("--mean", type=float, default=0.0, help="高斯噪声均值（归一化域）")
    parser.add_argument(
        "--variances",
        type=str,
        default="0.01,0.02,0.05",
        help="噪声方差列表，逗号分隔，如 0.01,0.02,0.05",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variances = parse_variances(args.variances)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])

    if not image_paths:
        raise FileNotFoundError(f"在 {input_dir} 未找到图像文件")

    for img_path in image_paths:
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue

        for var in variances:
            noisy = add_gaussian_noise(image=image, mean=args.mean, var=var, seed=args.seed)
            out_name = f"{img_path.stem}_gauss_mean{args.mean:.2f}_var{var:.3f}{img_path.suffix.lower()}"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), noisy)
            print(f"[保存] {out_path}")


if __name__ == "__main__":
    main()
