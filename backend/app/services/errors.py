"""后端服务层的可控错误。"""


class BackendServiceError(RuntimeError):
    """可安全返回给API调用方的业务错误。"""

    def __init__(self, message: str, *, code: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class InvalidUploadError(BackendServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="invalid_upload", status_code=400)


class CaseNotFoundError(BackendServiceError):
    def __init__(self, case_id: str) -> None:
        super().__init__(
            f"病例不存在：{case_id}",
            code="case_not_found",
            status_code=404,
        )


class CaseConflictError(BackendServiceError):
    def __init__(self, case_id: str) -> None:
        super().__init__(
            f"病例已存在：{case_id}",
            code="case_conflict",
            status_code=409,
        )


class CaseStateError(BackendServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="invalid_case_state", status_code=409)


class PipelineExecutionError(BackendServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="analysis_failed", status_code=500)


class AnalysisTaskNotFoundError(BackendServiceError):
    def __init__(self, task_id: str) -> None:
        super().__init__(
            f"分析任务不存在：{task_id}",
            code="analysis_task_not_found",
            status_code=404,
        )


class ComparisonNotFoundError(BackendServiceError):
    def __init__(self, comparison_id: str) -> None:
        super().__init__(
            f"随访对比不存在：{comparison_id}",
            code="comparison_not_found",
            status_code=404,
        )


class ComparisonTaskNotFoundError(BackendServiceError):
    def __init__(self, task_id: str) -> None:
        super().__init__(
            f"空间对比任务不存在：{task_id}",
            code="comparison_task_not_found",
            status_code=404,
        )


class TaskQueueUnavailableError(BackendServiceError):
    def __init__(self) -> None:
        super().__init__(
            "分析任务队列暂不可用，请确认Redis和Celery配置",
            code="task_queue_unavailable",
            status_code=503,
        )


class ExternalServiceError(BackendServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="external_service_failed", status_code=502)


class ServiceConfigurationError(BackendServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="service_not_configured", status_code=503)
