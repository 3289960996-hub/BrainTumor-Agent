"""BraTS 2021模态、标签和nnU-Net通道定义。"""

from enum import IntEnum, StrEnum


class MRIModality(StrEnum):
    """分割模型要求的四种MRI模态。"""

    T1 = "t1"
    T1CE = "t1ce"
    T2 = "t2"
    FLAIR = "flair"


class BraTSLabel(IntEnum):
    """BraTS原始分割标签值。"""

    BACKGROUND = 0
    NECROTIC_CORE = 1
    EDEMA = 2
    ENHANCING_TUMOR = 4


REQUIRED_MODALITIES: tuple[MRIModality, ...] = (
    MRIModality.T1,
    MRIModality.T1CE,
    MRIModality.T2,
    MRIModality.FLAIR,
)

# 固定通道顺序非常重要：训练、离线处理和在线推理必须使用同一映射。
NNUNET_CHANNELS: dict[MRIModality, str] = {
    MRIModality.T1: "0000",
    MRIModality.T1CE: "0001",
    MRIModality.T2: "0002",
    MRIModality.FLAIR: "0003",
}

BRATS_LABELS: dict[str, tuple[BraTSLabel, ...]] = {
    "ET": (BraTSLabel.ENHANCING_TUMOR,),
    "TC": (BraTSLabel.NECROTIC_CORE, BraTSLabel.ENHANCING_TUMOR),
    "WT": (
        BraTSLabel.NECROTIC_CORE,
        BraTSLabel.EDEMA,
        BraTSLabel.ENHANCING_TUMOR,
    ),
}
