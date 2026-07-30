"""后端在线推理服务边界。

离线数据准备、训练、批量推理与评估已经在segmentation脚本中实现；此服务
将在后续后端任务队列阶段接入GPU Worker，当前不在FastAPI进程内直接推理。
"""

from pathlib import Path

from segmentation.schemas import SegmentationRequest, SegmentationResult


class NnUNetSegmentationService:
    """已发布nnU-Net模型的异步GPU Worker适配器。"""

    def __init__(self, model_root: Path) -> None:
        self.model_root = model_root

    def predict(self, request: SegmentationRequest) -> SegmentationResult:
        """对单个病例运行在线推理。

        在线Worker集成尚未实现；批量推理请使用segmentation/inference.py。
        """

        raise NotImplementedError(
            "在线GPU Worker尚未接入；请先使用segmentation.inference批量推理"
        )
