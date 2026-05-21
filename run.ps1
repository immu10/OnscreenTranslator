# PowerShell equivalent of run.sh
# Runs main.py, prefixes each line with a timestamp, and writes to logs\run-*.log.
#
# Notes:
# - We do NOT use PowerShell's `2>&1` directly on python.exe, because Windows
#   PowerShell 5.1 wraps native stderr in ErrorRecords (NativeCommandError),
#   which trips on every progress-bar line. We let cmd.exe do the stderr merge
#   so PowerShell only sees one combined stdout stream.

if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = "logs\run-$stamp.log"

cmd /c "python -u main.py 2>&1" | ForEach-Object {
    "{0} {1}" -f (Get-Date -Format "HH:mm:ss.fff"), $_
} | Tee-Object -FilePath $log
