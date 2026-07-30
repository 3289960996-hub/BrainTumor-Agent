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


class ExternalServiceError(BackendServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="external_service_failed", status_code=502)


class ServiceConfigurationError(BackendServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="service_not_configured", status_code=503)
