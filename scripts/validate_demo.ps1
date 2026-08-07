param(
    [string]$ApiBase = "http://127.0.0.1:8000/api/v1",
    [string]$DemoCaseId = "case-031412bee7e3491c"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

function Require-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        throw $Message
    }
}

Write-Host "Checking backend health..."
$health = Invoke-RestMethod -Uri "$ApiBase/health" -TimeoutSec 15
Require-True ($health.status -eq "ok") "Backend health check failed."

foreach ($name in @("upload", "analysis", "report", "chat", "rag")) {
    Require-True ([bool]$health.capabilities.$name) "Capability is not ready: $name"
}

Write-Host "Checking prepared demo case: $DemoCaseId"
$case = Invoke-RestMethod -Uri "$ApiBase/cases/$DemoCaseId" -TimeoutSec 30
Require-True ($null -ne $case.mask) "Demo case has no segmentation mask."
Require-True ($null -ne $case.tumor_metrics) "Demo case has no quantitative metrics."
Require-True (-not [string]::IsNullOrWhiteSpace($case.report)) "Demo case has no report."

$metadataPath = Join-Path $projectRoot "runtime\data\cases\$DemoCaseId\case.json"
Require-True (Test-Path -LiteralPath $metadataPath) "Demo case metadata was not found."
$metadata = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
Require-True (-not [bool]$metadata.report_stale) "Demo case report is marked stale."

Write-Host "Demo validation passed."
Write-Host "Case: $DemoCaseId"
Write-Host "WT volume: $($case.tumor_metrics.tumor_volume) cm3"
Write-Host "Open: http://127.0.0.1:5173/?case=$DemoCaseId#workbench"
