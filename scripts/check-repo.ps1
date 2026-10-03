$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$failures = [System.Collections.Generic.List[string]]::new()

Write-Host "Checking nested Git repositories..."
$nestedGit = Get-ChildItem -Path $root -Directory -Force -Recurse -Filter ".git" |
    Where-Object { $_.FullName -ne (Join-Path $root ".git") }
if ($nestedGit) {
    $failures.Add("Nested .git directories: $($nestedGit.FullName -join ', ')")
}

Write-Host "Checking large files (>50 MiB)..."
$largeFiles = Get-ChildItem -Path $root -File -Recurse -Force |
    Where-Object {
        $_.FullName -notmatch "[\\/](\.git|\.venv|venv|__pycache__|data|outputs|checkpoints|artifacts|runs|wandb)[\\/]" -and
        $_.Length -gt 50MB
    }
if ($largeFiles) {
    $failures.Add("Large files: $($largeFiles.FullName -join ', ')")
}

Write-Host "Checking common secret filenames..."
$secretNames = @(".env", "credentials.json", "id_rsa", "id_ed25519")
$secretFiles = Get-ChildItem -Path $root -File -Recurse -Force |
    Where-Object {
        $_.FullName -notmatch "[\\/](\.git|\.venv|venv|__pycache__|data|outputs|checkpoints|artifacts|runs|wandb)[\\/]" -and
        ($secretNames -contains $_.Name -or $_.Extension -in @(".pem", ".key"))
    }
if ($secretFiles) {
    $failures.Add("Possible secret files: $($secretFiles.FullName -join ', ')")
}

Write-Host "Checking upstream licenses..."
$assignmentDirs = Get-ChildItem -Path (Join-Path $root "assignments") -Directory -Recurse |
    Where-Object { $_.Name -like "assignment*-*" }
foreach ($directory in $assignmentDirs) {
    if (-not (Test-Path (Join-Path $directory.FullName "LICENSE"))) {
        $readme = Join-Path $directory.FullName "README.md"
        $isSourceOnlyNotice = (Test-Path $readme) -and
            ((Get-Content $readme -Raw) -match "SOURCE_ONLY_NOTICE")
        if (-not $isSourceOnlyNotice) {
            $failures.Add("Missing upstream LICENSE: $($directory.FullName)")
        }
    }
}

if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Error $_ }
    exit 1
}

Write-Host "Repository checks passed."
