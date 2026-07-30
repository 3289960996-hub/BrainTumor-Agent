"""NIfTI读取和保存工具。

Nibabel负责体素数组与NIfTI头信息；SimpleITK独立读取尺寸、spacing、
origin和direction，用于减少单一读取器带来的几何解释风险。
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

from data_process.exceptions import NiftiReadError
from data_process.schemas import ImageGeometry, NiftiVolume


def is_nifti_path(path: Path) -> bool:
    """判断路径是否为支持的NIfTI文件。"""

    lower_name = path.name.lower()
    return lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")


def read_nifti(path: str | Path) -> NiftiVolume:
    """读取一个三维NIfTI文件并返回float32体素数据。

    读取时会检查：
    1. 文件扩展名和文件存在性；
    2. 数据是否为三维；
    3. 数据是否包含NaN或Inf；
    4. Nibabel与SimpleITK读取到的尺寸和spacing是否一致。
    """

    nifti_path = Path(path).expanduser().resolve()
    if not nifti_path.is_file():
        raise NiftiReadError(f"NIfTI文件不存在：{nifti_path}")
    if not is_nifti_path(nifti_path):
        raise NiftiReadError(f"仅支持.nii或.nii.gz文件：{nifti_path}")

    try:
        nib_image = nib.load(str(nifti_path))
        data = np.asarray(nib_image.get_fdata(dtype=np.float32), dtype=np.float32)
    except Exception as exc:
        raise NiftiReadError(f"Nibabel读取失败：{nifti_path}") from exc

    if data.ndim != 3:
        raise NiftiReadError(
            f"仅支持三维MRI，实际维度为{data.ndim}，文件：{nifti_path}"
        )
    if not np.isfinite(data).all():
        raise NiftiReadError(f"NIfTI包含NaN或Inf：{nifti_path}")

    try:
        sitk_image = sitk.ReadImage(str(nifti_path))
    except Exception as exc:
        raise NiftiReadError(f"SimpleITK读取失败：{nifti_path}") from exc

    sitk_shape = tuple(int(value) for value in sitk_image.GetSize())
    sitk_spacing = tuple(float(value) for value in sitk_image.GetSpacing())
    nib_spacing = tuple(float(value) for value in nib_image.header.get_zooms()[:3])

    if tuple(data.shape) != sitk_shape:
        raise NiftiReadError(
            "Nibabel与SimpleITK读取到的尺寸不一致："
            f"nibabel={tuple(data.shape)}，SimpleITK={sitk_shape}，文件={nifti_path}"
        )
    if not np.allclose(nib_spacing, sitk_spacing, rtol=0.0, atol=1e-5):
        raise NiftiReadError(
            "Nibabel与SimpleITK读取到的spacing不一致："
            f"nibabel={nib_spacing}，SimpleITK={sitk_spacing}，文件={nifti_path}"
        )

    geometry = ImageGeometry(
        shape=sitk_shape,
        spacing=sitk_spacing,
        origin=tuple(float(value) for value in sitk_image.GetOrigin()),
        direction=tuple(float(value) for value in sitk_image.GetDirection()),
        affine=np.asarray(nib_image.affine, dtype=np.float64),
    )
    return NiftiVolume(
        path=nifti_path,
        data=data,
        geometry=geometry,
        header=nib_image.header.copy(),
    )


def save_nifti(
    data: NDArray[np.floating],
    reference: NiftiVolume,
    output_path: str | Path,
) -> Path:
    """使用参考影像的affine和头信息保存float32 NIfTI。"""

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(data, dtype=np.float32)
    if array.shape != reference.geometry.shape:
        raise ValueError(
            f"待保存数据尺寸{array.shape}与参考尺寸{reference.geometry.shape}不一致"
        )
    if not np.isfinite(array).all():
        raise ValueError("待保存数据包含NaN或Inf")

    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    output_image = nib.Nifti1Image(array, reference.geometry.affine, header=header)

    # 保留qform/sform编码，确保后续nnU-Net和阅片工具使用相同物理空间。
    qform_code = int(reference.header["qform_code"]) or 1
    sform_code = int(reference.header["sform_code"]) or 1
    output_image.set_qform(reference.geometry.affine, code=qform_code)
    output_image.set_sform(reference.geometry.affine, code=sform_code)
    nib.save(output_image, str(target))
    return target
