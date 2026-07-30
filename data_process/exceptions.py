"""MRI数据处理模块的领域异常。"""


class DataProcessError(RuntimeError):
    """所有数据处理异常的基类。"""


class DatasetDiscoveryError(DataProcessError):
    """病例目录结构或模态文件命名不符合要求。"""


class NiftiReadError(DataProcessError):
    """NIfTI文件无法读取或内容非法。"""


class ManifestValidationError(DataProcessError):
    """病例清单不完整或包含重复模态。"""


class GeometryMismatchError(DataProcessError):
    """四模态尺寸、spacing或空间变换不一致。"""


class NormalizationError(DataProcessError):
    """MRI强度归一化失败。"""


class ProcessedDataExistsError(DataProcessError):
    """目标processed文件已存在且未允许覆盖。"""


class VisualizationError(DataProcessError):
    """MRI或分割掩膜不满足可视化要求。"""
