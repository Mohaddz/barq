param(
    [string]$Downloads = (Join-Path $env:USERPROFILE 'Downloads')
)

$ErrorActionPreference = 'Stop'
for ($i = 6; $i -le 9; $i++) {
    $filename = "dev($i).zip"
    $source = Join-Path $Downloads $filename
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required saved BALSAM archive not found: $source"
    }
    & modal volume put barq-data $source "/benchmarks/balsam-v2-saved/$filename"
    if ($LASTEXITCODE -ne 0) { throw "Modal upload failed for $filename" }
}
Write-Output 'Staged dev(6).zip through dev(9).zip on Modal volume barq-data.'
