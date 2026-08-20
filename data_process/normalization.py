"""低内存MRI强度归一化。"""

from dataclasses import dataclass

import numpy as np
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
    empty_channels: list[str] = []
    for index, channel in enumerate(data):
        if not np.isfinite(channel).all():
            raise NormalizationError("归一化输入包含NaN或Inf")
        if not np.any(channel != 0):
            empty_channels.append(REQUIRED_MODALITIES[index].value)
    if empty_channels:
        raise NormalizationError(f"模态数据全为0：{', '.join(empty_channels)}")

    result = np.zeros_like(data, dtype=np.float32)
    if resolved_config.channel_wise:
        for source, target in zip(data, result, strict=True):
            _normalize_array(source, target, nonzero=resolved_config.nonzero)
    else:
        _normalize_array(data, result, nonzero=resolved_config.nonzero)
    return result


def _normalize_array(
    source: NDArray[np.float32],
    target: NDArray[np.float32],
    *,
    nonzero: bool,
) -> None:
    mask = source != 0 if nonzero else np.ones(source.shape, dtype=np.bool_)
    mean = np.mean(source, where=mask, dtype=np.float32)
    std = np.std(source, where=mask, dtype=np.float32)
    if not np.isfinite(mean) or not np.isfinite(std):
        raise NormalizationError("归一化统计量包含NaN或Inf")
    if std == 0:
        std = np.float32(1.0)
    np.subtract(source, mean, out=target, where=mask)
    np.divide(target, std, out=target, where=mask)
