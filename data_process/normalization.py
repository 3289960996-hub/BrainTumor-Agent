"""基于MONAI的MRI强度归一化。"""

from dataclasses import dataclass

import numpy as np
from monai.transforms import NormalizeIntensity
from numpy.typing import NDArray

from data_process.constants import REQUIRED_MODALITIES
from data_process.exceptions import NormalizationError


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """MRI归一化参数。

    BraTS影像背景通常为0，默认仅在非零脑组织区域内逐模态计算均值和标准差，
    可避免大面积背景影响统计量。
    """

    nonzero: bool = True
    channel_wise: bool = True


def normalize_multimodal(
    stacked_data: NDArray[np.floating],
    config: NormalizationConfig | None = None,
) -> NDArray[np.float32]:
    """将(C, X, Y, Z)四模态MRI执行逐通道Z-score归一化。"""

    resolved_config = config or NormalizationConfig()
    data = np.asarray(stacked_data, dtype=np.float32)

    if data.ndim != 4:
        raise NormalizationError(
            f"归一化输入必须为(C, X, Y, Z)四维数组，实际shape={data.shape}"
        )
    if data.shape[0] != len(REQUIRED_MODALITIES):
        raise NormalizationError(
            f"归一化输入必须包含4个模态通道，实际通道数={data.shape[0]}"
        )
    if not np.isfinite(data).all():
        raise NormalizationError("归一化输入包含NaN或Inf")

    # 全零模态通常意味着文件错误或上传错误，应在进入模型前明确拒绝。
    empty_channels = [
        REQUIRED_MODALITIES[index].value
        for index, channel in enumerate(data)
        if not np.any(channel != 0)
    ]
    if empty_channels:
        raise NormalizationError(f"模态数据全为0：{', '.join(empty_channels)}")

    transform = NormalizeIntensity(
        nonzero=resolved_config.nonzero,
        channel_wise=resolved_config.channel_wise,
        dtype=np.float32,
    )
    try:
        normalized = transform(data)
    except Exception as exc:
        raise NormalizationError("MONAI强度归一化失败") from exc

    # MONAI可能返回MetaTensor；显式转为CPU NumPy，便于NIfTI保存。
    if hasattr(normalized, "detach"):
        normalized = normalized.detach().cpu().numpy()
    result = np.asarray(normalized, dtype=np.float32)

    if not np.isfinite(result).all():
        raise NormalizationError("归一化结果包含NaN或Inf")
    return result
