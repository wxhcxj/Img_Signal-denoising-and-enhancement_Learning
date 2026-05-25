# Img_Signal-denoising-and-enhancement_Learning

> 图像信号去噪和增强学习项目。  
> 课程设计课题：**课题4《信号降噪与增强》**。

本项目围绕图像信号去噪任务，分别实现并对比传统滤波方法与深度学习方法，包括：

- 维纳滤波（Wiener Filter）
- 卡尔曼滤波（Kalman Filter）
- 去噪自编码器（Denoising Autoencoder, DAE）
- U-Net 去噪模型

项目目标是对干净图像添加不同强度的高斯噪声，再使用不同算法进行去噪，最后通过 **PSNR** 和 **SSIM** 等指标进行统一评价和对比分析。

---

## 1. 项目任务定义

### 输入

- 原始干净图像 `clean image`
- 添加高斯噪声后的图像 `noisy image`

### 方法

- 传统方法：Wiener、Kalman
- 深度学习方法：DAE、U-Net

### 输出

- 各算法去噪后的图像
- PSNR / SSIM 指标结果
- Loss 曲线、实验对比图、指标表格
- 最终报告与答辩 PPT 所需结果材料

### 评价指标

- **PSNR**：峰值信噪比，用于衡量去噪图像与原始干净图像之间的像素误差。
- **SSIM**：结构相似性，用于衡量去噪图像在亮度、对比度和结构上的保真程度。

---

## 2. 推荐环境

建议使用如下环境：

```bash
Python >= 3.10
PyTorch
OpenCV
NumPy
Pandas
Matplotlib
scikit-image
```

建议创建独立虚拟环境：

```bash
conda create -n img-denoise python=3.10
conda activate img-denoise
```

安装依赖示例：

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib scikit-image tqdm
```

如果项目中已经整理了 `requirements.txt`，也可以直接执行：

```bash
pip install -r requirements.txt
```

---

## 3. 项目目录结构

项目目录建议保持如下结构：

```text
Img_Signal-denoising-and-enhancement_Learning/
├── data/
│   ├── BSR_bsds500/              # 深度学习使用的原始数据集
│   └── denoised/                 # 深度学习方法输出的去噪结果
│
├── data_T/
│   ├── raw/                      # 传统方法使用的原始图像
│   ├── noisy/                    # 传统方法使用的加噪图像
│   └── denoised/                 # 传统方法输出的去噪结果
│
├── DLdata/                       # 深度学习预处理后的数据
│
├── results/
│   ├── checkpoints/              # 模型权重
│   ├── figures/                  # Loss 曲线、可视化图
│   └── metrics/                  # 每轮 loss、PSNR、SSIM 等指标
│
├── src/
│   ├── deep/
│   │   ├── dataset.py            # 深度学习数据读取与预处理
│   │   ├── dae_model.py          # DAE 模型
│   │   ├── train_dae.py          # DAE 训练
│   │   ├── infer_dae.py          # DAE 推理
│   │   ├── unet_model.py         # U-Net 模型
│   │   ├── train_unet.py         # U-Net 训练
│   │   └── infer_unet.py         # U-Net 推理
│   │
│   ├── traditional/
│   │   ├── wiener_denoise.py     # 维纳滤波去噪
│   │   └── kalman_denoise.py     # 卡尔曼滤波去噪
│   │
│   └── utils/
│       ├── add_gaussian_noise.py # 添加高斯噪声
│       ├── evaluate.py           # 计算 PSNR / SSIM
│       └── seed.py               # 固定随机种子
│
└── README.md
```

---

## 4. 完整工作流

推荐按照以下顺序执行整个项目。

---

### 4.1 固定随机种子

为了保证实验可复现，先检查或运行随机种子脚本。

```bash
python src/utils/seed.py
```

如果在服务器环境中使用绝对路径，也可以按自己的项目路径执行，例如：

```bash
python /data/lzh/Project/Img_Signal-denoising-and-enhancement_Learning-main/src/utils/seed.py
```

---

### 4.2 准备传统方法数据

传统方法使用 `data_T` 目录。

推荐从 SIPI 图像数据库中选择图像：

- 2 张灰度图
- 2 张彩色图

原始图像放入：

```text
data_T/raw/
```

---

### 4.3 添加高斯噪声

执行高斯噪声生成脚本：

```bash
python src/utils/add_gaussian_noise.py \
  --input_dir data_T/raw \
  --output_dir data_T/noisy \
  --mean 0 \
  --variances 0.01 \
  --seed 42
```

也可以测试多档噪声：

```bash
python src/utils/add_gaussian_noise.py \
  --input_dir data_T/raw \
  --output_dir data_T/noisy \
  --mean 0 \
  --variances 0.01 0.02 0.05 \
  --seed 42
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--mean 0` | 噪声均值为 0，表示噪声不会整体抬高或降低图像亮度 |
| `--variances 0.01` | 噪声方差，控制噪声强度 |
| `--seed 42` | 固定随机种子，保证结果可复现 |

噪声强度建议：

| 方差 | 噪声等级 | 说明 |
|---|---|---|
| `0.01` | 轻度噪声 | 图像细节仍较清晰 |
| `0.02` | 中等噪声 | 噪声明显，适合做主要对比 |
| `0.05` | 较强噪声 | 去噪难度更高，可测试算法鲁棒性 |

---

## 5. 传统方法实验

---

### 5.1 维纳滤波 Wiener Filter

维纳滤波基于局部统计量进行去噪。每个像素会参考周围窗口中的局部均值和局部方差，从而估计更干净的像素值。

手动指定噪声功率：

```bash
python src/traditional/wiener_denoise.py \
  --input_dir data_T/noisy/mean0_var001 \
  --output_dir data_T/denoised/wiener/mean0_var001 \
  --kernel_size 5 \
  --noise_var 0.01
```

也可以由算法估计噪声功率：

```bash
python src/traditional/wiener_denoise.py \
  --input_dir path_to_noisy \
  --output_dir path_to_denoised \
  --kernel_size 5
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--kernel_size 5` | 局部窗口大小为 `5×5` |
| `--noise_var 0.01` | 手动指定噪声方差 |
| `--input_dir` | 输入加噪图像目录 |
| `--output_dir` | 输出去噪图像目录 |

`kernel_size` 调参建议：

| 窗口大小 | 特点 |
|---|---|
| `3` | 保留细节较好，但去噪能力偏弱 |
| `5` | 折中选择，推荐起步参数 |
| `7` / `9` | 去噪更强，但可能模糊边缘和纹理 |

---

### 5.2 卡尔曼滤波 Kalman Filter

推荐起步配置：逐行建模 + 中等噪声设定。

```bash
python src/traditional/kalman_denoise.py \
  --mode row \
  --q 1e-4 \
  --r 0.01 \
  --input_dir data_T/noisy/mean0_var001 \
  --output_dir data_T/denoised/kalman/mean0_var001
```

逐列建模：

```bash
python src/traditional/kalman_denoise.py \
  --mode col \
  --q 1e-4 \
  --r 0.01
```

像素序列化建模：

```bash
python src/traditional/kalman_denoise.py \
  --mode raster \
  --q 1e-4 \
  --r 0.01
```

参数说明：

| 参数 | 含义 | 建议 |
|---|---|---|
| `--mode` | 图像建模方式 | `row` / `col` / `raster` |
| `--q` | 过程噪声 Q | 建议扫 `1e-5, 1e-4, 1e-3, 1e-2` |
| `--r` | 观测噪声 R | 与加噪方差对齐，如 `0.01 / 0.02 / 0.05` |
| `--a` | 状态转移系数 A | 默认 `1`，后期可在 `0.95~1.05` 微调 |
| `--h` | 观测系数 H | 默认 `1`，一般先固定 |
| `--p0` | 初始协方差 P0 | 默认 `1`，建议范围 `0.1~10` |

`mode` 对比：

| 模式 | 说明 |
|---|---|
| `row` | 逐行处理，更利用横向连续性，通常是最好起点 |
| `col` | 逐列处理，对竖向结构更友好 |
| `raster` | 像素序列化，实现简单，但空间结构利用偏弱，适合作为对照组 |

`q` 和 `r` 的影响：

| 参数变化 | 效果 |
|---|---|
| `q` 小 | 输出更平滑，抑噪更强，但细节可能被抹除 |
| `q` 大 | 更跟随观测，细节保留较多，但噪声残留增大 |
| `r` 大 | 更不信任观测图像，输出更平滑 |
| `r` 小 | 更信任观测图像，输出更接近原噪声图 |

---

## 6. 深度学习方法实验

深度学习部分主要使用：

- `data/BSR_bsds500/`：原始数据集
- `DLdata/`：预处理后的训练数据
- `results/checkpoints/`：保存模型权重
- `results/figures/`：保存训练曲线
- `results/metrics/`：保存训练和推理指标

当前样例主要使用 `var=0.01`，后续建议补充 `var=0.02` 和 `var=0.05`，用于测试模型在不同噪声强度下的鲁棒性。

---

### 6.1 数据集预处理

执行数据处理脚本：

```bash
python -m src.deep.dataset --var 0.01
```

如需生成其他噪声等级的数据：

```bash
python -m src.deep.dataset --var 0.02
python -m src.deep.dataset --var 0.05
```

---

### 6.2 检查模型结构

检查 DAE 模型：

```bash
python -m src.deep.dae_model
```

检查 U-Net 模型：

```bash
python -m src.deep.unet_model
```

该步骤可用于查看模型是否能正常构建，以及参数量是否符合预期。

---

## 7. DAE 去噪自编码器

DAE 是一种编码器—解码器结构。模型先将带噪图像压缩为深层特征，再通过解码器恢复出干净图像。

基本思想：

```text
Noisy Image -> Encoder -> Latent Feature -> Decoder -> Denoised Image
```

优点：

- 结构简单
- 容易训练
- 适合作为深度学习去噪基线

不足：

- 多次下采样可能导致边缘、纹理等细节信息丢失
- 对复杂图像细节恢复能力弱于 U-Net

---

### 7.1 训练 DAE

```bash
python -m src.deep.train_dae --var 0.01
```

后续建议继续补充：

```bash
python -m src.deep.train_dae --var 0.02
python -m src.deep.train_dae --var 0.05
```

当前记录中，DAE 样本最优结果大约出现在 400 轮附近。训练完成后，建议将 checkpoint 文件名中补充轮次信息，例如：

```text
dae_best_400.pth
```

---

### 7.2 DAE 推理

指定模型和噪声方差：

```bash
python -m src.deep.infer_dae \
  --checkpoint results/checkpoints/dae_best_400.pth \
  --var 0.01
```

推理输出建议带有时间戳和配置名，例如包含：

```text
checkpoint 名称
var 噪声方差
推理时间戳
```

这样可以避免不同实验结果互相覆盖。

---

## 8. U-Net 去噪模型

U-Net 是经典的编码器—解码器结构，最早常用于医学图像分割。由于它能够很好地保留图像细节，也适合用于图像去噪、图像增强和图像恢复任务。

U-Net 相比 DAE 的关键改进是加入了 **跳跃连接 Skip Connection**。

其核心思想是：

- 深层特征：理解图像整体结构
- 浅层特征：保留边缘、纹理、细节
- 跳跃连接：将编码器浅层特征直接传给解码器对应层

基本结构：

```text
Noisy Image
    ↓
Encoder ─────────────┐
    ↓                │ Skip Connection
Bottleneck           │
    ↓                │
Decoder <────────────┘
    ↓
Denoised Image
```

优点：

- 细节保留能力强
- 对边缘和纹理恢复更友好
- 通常比普通 DAE 有更好的视觉效果和指标表现

---

### 8.1 训练 U-Net

```bash
python -m src.deep.train_unet --var 0.01 --epochs 1000
```

如需训练其他噪声方差：

```bash
python -m src.deep.train_unet --var 0.02 --epochs 1000
python -m src.deep.train_unet --var 0.05 --epochs 1000
```

当前样例设置了 1000 轮，Loss 曲线基本下降到较稳定状态。后续如果换用更大的数据集，例如 Waterloo Exploration Database，需要根据实际 Loss 曲线重新设置训练轮数。

训练完成后建议将 checkpoint 文件名中补充轮次信息，例如：

```text
unet_best_1000.pth
```

---

### 8.2 U-Net 推理

```bash
python -m src.deep.infer_unet \
  --checkpoint results/checkpoints/unet_best_1000.pth \
  --data_root DLdata \
  --output_dir data/denoised/unet \
  --metrics_csv results/metrics/unet_infer_metrics.csv \
  --var 0.01
```

如需推理其他噪声方差，修改 `--var`，并确认 checkpoint 与数据方差匹配。

---

## 9. 统一评价 PSNR / SSIM

传统方法和深度学习方法完成推理后，需要统一计算 PSNR 和 SSIM。

如果 `evaluate.py` 已经配置好默认路径，可以直接运行：

```bash
python src/utils/evaluate.py
```

建议统一输出到：

```text
results/metrics/
```

建议最终整理成如下表格：

| 方法 | 噪声方差 | PSNR | SSIM | 备注 |
|---|---:|---:|---:|---|
| Noisy | 0.01 | - | - | 加噪图像基线 |
| Wiener | 0.01 | - | - | 传统滤波 |
| Kalman-row | 0.01 | - | - | 逐行建模 |
| Kalman-col | 0.01 | - | - | 逐列建模 |
| Kalman-raster | 0.01 | - | - | 像素序列化 |
| DAE | 0.01 | - | - | 深度学习基线 |
| U-Net | 0.01 | - | - | 深度学习增强模型 |

最终建议分别统计三档噪声：

```text
var = 0.01
var = 0.02
var = 0.05
```

---

## 10. 推荐实验流程总览

### 第一步：传统方法数据准备

```bash
python src/utils/add_gaussian_noise.py \
  --input_dir data_T/raw \
  --output_dir data_T/noisy \
  --mean 0 \
  --variances 0.01 0.02 0.05 \
  --seed 42
```

### 第二步：运行 Wiener

```bash
python src/traditional/wiener_denoise.py \
  --input_dir data_T/noisy/mean0_var001 \
  --output_dir data_T/denoised/wiener/mean0_var001 \
  --kernel_size 5 \
  --noise_var 0.01
```

### 第三步：运行 Kalman

```bash
python src/traditional/kalman_denoise.py \
  --mode row \
  --q 1e-4 \
  --r 0.01 \
  --input_dir data_T/noisy/mean0_var001 \
  --output_dir data_T/denoised/kalman/mean0_var001
```

### 第四步：深度学习数据预处理

```bash
python -m src.deep.dataset --var 0.01
```

### 第五步：训练 DAE

```bash
python -m src.deep.train_dae --var 0.01
```

### 第六步：DAE 推理

```bash
python -m src.deep.infer_dae \
  --checkpoint results/checkpoints/dae_best_400.pth \
  --var 0.01
```

### 第七步：训练 U-Net

```bash
python -m src.deep.train_unet --var 0.01 --epochs 1000
```

### 第八步：U-Net 推理

```bash
python -m src.deep.infer_unet \
  --checkpoint results/checkpoints/unet_best_1000.pth \
  --data_root DLdata \
  --output_dir data/denoised/unet \
  --metrics_csv results/metrics/unet_infer_metrics.csv \
  --var 0.01
```

### 第九步：统一评价

```bash
python src/utils/evaluate.py
```

---

## 11. 调参建议

---

### 11.1 Wiener 调参

优先调整：

```text
kernel_size = 3 / 5 / 7 / 9
noise_var = 0.01 / 0.02 / 0.05
```

建议：

- 从 `kernel_size=5` 开始。
- 噪声越强，可以适当增大窗口。
- 如果图像边缘被明显抹平，说明窗口可能过大。

---

### 11.2 Kalman 调参

优先扫描：

```text
mode = row / col / raster
q = 1e-5 / 1e-4 / 1e-3 / 1e-2
r = 0.01 / 0.02 / 0.05
```

建议：

- `mode=row` 作为默认起点。
- `r` 与加噪方差保持一致。
- `q` 越小，图像越平滑；`q` 越大，细节更多但噪声残留也更多。

---

### 11.3 DAE 调参

建议关注：

```text
var
batch_size
learning_rate
epochs
loss function
```

说明：

- `var=0.01` 可以作为入门实验。
- 后续需要补充 `0.02` 和 `0.05`。
- 如果训练 Loss 下降后不再改善，可以停止训练或降低学习率。

---

### 11.4 U-Net 调参

建议关注：

```text
var
epochs
batch_size
learning_rate
checkpoint selection
```

说明：

- 当前样例使用 `epochs=1000`。
- 如果更换为更大的数据集，需要根据 Loss 曲线决定训练轮数。
- 推理时要确认 checkpoint 与 `--var` 对应。

---

## 12. 可视化与报告建议

最终报告或 PPT 中建议包含：

1. 项目任务定义
2. 数据集和噪声设置
3. 高斯噪声参数说明
4. Wiener 原理与参数分析
5. Kalman 原理与参数分析
6. DAE 网络结构与训练曲线
7. U-Net 网络结构与跳跃连接说明
8. PSNR / SSIM 对比表
9. 去噪效果可视化对比图
10. 不同噪声强度下的鲁棒性分析
11. 结论与改进方向

推荐可视化拼图格式：

```text
Clean | Noisy | Wiener | Kalman | DAE | U-Net
```

如果空间允许，可以增加局部放大图，用来比较边缘、纹理、细节区域的恢复效果。

---

## 13. 当前已完成内容

根据当前工作记录，项目已经完成或规划了以下内容：

- [x] 新建项目 `Img_Signal-denoising-and-enhancement_Learning`
- [x] 设计项目目录结构
- [x] 创建 `seed.py` 固定随机种子
- [x] 增加高斯噪声添加脚本
- [x] 增加 Wiener 维纳滤波模块
- [x] 增加 Kalman 卡尔曼滤波模块
- [x] 增加 PSNR / SSIM 评价模块
- [x] 增加 `dataset.py` 数据读取与预处理模块
- [x] 增加 DAE 模型
- [x] 增加 DAE 训练脚本
- [x] 增加 DAE 推理脚本
- [x] 增加 U-Net 模型
- [x] 增加 U-Net 训练脚本
- [x] 增加 U-Net 推理脚本
- [x] 整理各模块启动命令

---

## 14. 后续待完成内容

- [ ] 补充 `var=0.02` 和 `var=0.05` 的完整实验
- [ ] 对 Wiener 的 `kernel_size` 进行多组对比
- [ ] 对 Kalman 的 `mode / q / r` 进行多组对比
- [ ] 整理 PSNR / SSIM 总表
- [ ] 绘制 DAE 和 U-Net 的 Loss 曲线
- [ ] 生成最终对比图：`clean | noisy | wiener | kalman | DAE | U-Net`
- [ ] 对局部细节区域做放大对比
- [ ] 完成课程设计报告
- [ ] 完成答辩 PPT
- [ ] 检查代码是否能从零运行，避免硬编码绝对路径

---

## 15. 最低可交付版本

如果时间有限，至少保证完成以下内容：

- [ ] 至少 1 张干净图像
- [ ] 至少 1 张高斯噪声图像
- [ ] Wiener、Kalman、DAE、U-Net 四种方法全部跑通
- [ ] 每种方法保存去噪结果图
- [ ] 输出 PSNR / SSIM 对比表
- [ ] 提交完整代码仓库
- [ ] 提交包含方法、结果和结论的实验报告

---

## 16. 注意事项

1. 所有实验应尽量使用同一批 noisy 输入图像，保证对比公平。
2. 计算 PSNR / SSIM 时必须使用同一套评价脚本，避免评价口径不一致。
3. 训练完成后建议重命名 checkpoint，例如：

```text
dae_best_400.pth
unet_best_1000.pth
```

4. 推理输出建议带上时间戳、checkpoint 名称和 var 参数，防止结果覆盖。
5. 不建议在最终代码中保留个人服务器绝对路径，应尽量改为相对路径。
6. `batch_size` 需要根据本机或服务器显存灵活设置。
7. 如果使用更大的 Waterloo Exploration Database，需要重新评估训练轮数和存储空间。

---

## 17. 参考实验命令汇总

```bash
# 1. 固定随机种子
python src/utils/seed.py

# 2. 添加高斯噪声
python src/utils/add_gaussian_noise.py \
  --input_dir data_T/raw \
  --output_dir data_T/noisy \
  --mean 0 \
  --variances 0.01 0.02 0.05 \
  --seed 42

# 3. Wiener 去噪
python src/traditional/wiener_denoise.py \
  --input_dir data_T/noisy/mean0_var001 \
  --output_dir data_T/denoised/wiener/mean0_var001 \
  --kernel_size 5 \
  --noise_var 0.01

# 4. Kalman 去噪
python src/traditional/kalman_denoise.py \
  --mode row \
  --q 1e-4 \
  --r 0.01 \
  --input_dir data_T/noisy/mean0_var001 \
  --output_dir data_T/denoised/kalman/mean0_var001

# 5. 深度学习数据预处理
python -m src.deep.dataset --var 0.01

# 6. 检查 DAE 模型
python -m src.deep.dae_model

# 7. 检查 U-Net 模型
python -m src.deep.unet_model

# 8. 训练 DAE
python -m src.deep.train_dae --var 0.01

# 9. DAE 推理
python -m src.deep.infer_dae \
  --checkpoint results/checkpoints/dae_best_400.pth \
  --var 0.01

# 10. 训练 U-Net
python -m src.deep.train_unet --var 0.01 --epochs 1000

# 11. U-Net 推理
python -m src.deep.infer_unet \
  --checkpoint results/checkpoints/unet_best_1000.pth \
  --data_root DLdata \
  --output_dir data/denoised/unet \
  --metrics_csv results/metrics/unet_infer_metrics.csv \
  --var 0.01

# 12. 统一评价
python src/utils/evaluate.py
```

---

## 18. 项目结论撰写方向

最终实验结论可以从以下角度展开：

- Wiener 方法实现简单、速度快，但对窗口大小和噪声估计较敏感。
- Kalman 方法可以通过状态空间建模实现平滑去噪，但不同建模方式和参数会明显影响效果。
- DAE 能够学习从噪声图到干净图的映射，但普通编码器—解码器结构可能损失部分细节。
- U-Net 通过跳跃连接保留浅层细节信息，通常在视觉效果和 SSIM 上更有优势。
- 随着噪声方差从 `0.01` 增加到 `0.05`，所有方法的去噪难度都会上升，深度学习方法的鲁棒性需要通过多噪声等级训练和测试进一步验证。
