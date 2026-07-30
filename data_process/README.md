# BraTS 2021 MRI数据处理模块

该模块只负责数据发现、读取、几何检查、强度归一化和processed数据保存，不包含模型训练。

## 输入目录

单个病例目录应包含：

- `BraTS2021_00000_t1.nii.gz`
- `BraTS2021_00000_t1ce.nii.gz`
- `BraTS2021_00000_t2.nii.gz`
- `BraTS2021_00000_flair.nii.gz`

四个文件必须具有相同的三维尺寸、spacing、affine、origin和direction。

## 处理规则

1. 使用Nibabel读取float32体素数组。
2. 使用SimpleITK独立读取并核对几何信息。
3. 按T1、T1ce、T2、FLAIR顺序堆叠为`(4, X, Y, Z)`。
4. 使用MONAI `NormalizeIntensity`在非零区域逐通道执行Z-score。
5. 保存为nnU-Net兼容通道文件及`metadata.json`。

该模块不会重采样或自动修复几何不一致的影像。几何不一致通常代表输入或配准错误，默认直接终止，避免静默改变医学空间。

## 命令行示例

在项目根目录运行：

`python -m data_process --input-dir D:\BraTS2021\BraTS2021_00000 --output-dir D:\BraTS2021_processed`

允许覆盖现有processed文件时增加：

`--overwrite`

## Python API示例

可运行示例文件：

`python -m data_process.example_usage --input-dir D:\BraTS2021\BraTS2021_00000 --output-dir D:\BraTS2021_processed`

核心调用入口为`BraTSDataProcessor.process_case`。

## 输出

输出目录包含：

- `病例ID_0000.nii.gz`：T1
- `病例ID_0001.nii.gz`：T1ce
- `病例ID_0002.nii.gz`：T2
- `病例ID_0003.nii.gz`：FLAIR
- `metadata.json`：源文件、通道映射、尺寸、spacing、affine和归一化配置

## 测试

测试使用临时生成的合成NIfTI，不需要BraTS真实数据或GPU：

`pytest data_process/tests -q`

## MRI可视化与AI勾画阅片

生成T1、T1ce、T2、FLAIR四窗口Matplotlib PNG和模拟放射科阅片界面的Plotly HTML：

`python -m data_process.visualize --input-dir D:\BraTS2021\BraTS2021_00000 --output-dir D:\BraTS_visualizations`

如果病例目录中存在`*_seg.nii.gz`会自动叠加。也可以显式指定：

`python -m data_process.visualize --input-dir D:\BraTS2021\BraTS2021_00000 --seg-path D:\masks\case_seg.nii.gz --axis axial --output-dir D:\BraTS_visualizations`

常用参数：

- `--axis axial|coronal|sagittal`：选择切片方向。
- `--slice-index 75`：指定静态图切片；省略时选择肿瘤面积最大层。
- `--slice-step 2`：Plotly每隔两层生成一帧，可减小HTML。
- `--mask-alpha 0.45`：设置mask透明度。
- `--initial-modality flair|t1|t1ce|t2`：设置交互阅片器的初始模态。
- `--label-space brats|nnunet`：查看恢复后的BraTS标签或nnU-Net内部标签。
- `--no-html`：只生成Matplotlib PNG。
- `--show`：在本机打开Matplotlib窗口。

交互HTML支持：

- T1、T1ce、T2、FLAIR模态下拉切换。
- 轴位/冠状位/矢状位切片滑块和自动播放；方向由`--axis`选择。
- Tumor Core、Edema、Enhancing Tumor半透明叠加。
- 显示全部区域、单独显示某一区域或隐藏AI勾画。
- Plotly缩放、平移、截图和滚轮缩放。

mask与MRI尺寸、spacing或affine不一致时会拒绝叠加，不执行隐式配准或重采样。
