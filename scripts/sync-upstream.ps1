param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$headers = @{ "User-Agent" = "Stanford-CS336-study-project" }

$snapshots = @(
    @{ Year = "spring2025"; Repo = "assignment1-basics"; Ref = "430e2c844e29f8aad8f9330e8706db9cb508241f" },
    @{ Year = "spring2025"; Repo = "assignment2-systems"; Ref = "e495ed740080661ab084914674d3e10048778889" },
    @{ Year = "spring2025"; Repo = "assignment3-scaling"; Ref = "09d205bde59e5c533368c0209dc86a4d5e4323ea" },
    @{ Year = "spring2025"; Repo = "assignment4-data"; Ref = "5a5f890cd9b72733e6273a596c40b696df3aa9df" },
    @{ Year = "spring2025"; Repo = "assignment5-alignment"; Ref = "a88071d1272112306f59257d946e43b56bb31773" },
    @{ Year = "spring2026"; Repo = "assignment1-basics"; Ref = "a158843b20107949f1a8d7df1b05cd33b9166712" },
    @{ Year = "spring2026"; Repo = "assignment2-systems"; Ref = "ca8bc81a59b70516f7ebb2da4808daade877c736" },
    @{ Year = "spring2026"; Repo = "assignment3-scaling"; Ref = "03e9372992e913061b9e78b5cfcb62ad8a87de35" },
    @{ Year = "spring2026"; Repo = "assignment4-data"; Ref = "0555bea66369872d912652debf10b115ca0688c8" }
)

# Spring 2026 Assignment 5 is deliberately not downloaded: the inspected
# upstream commit has no license. Its directory contains a source-only notice.
foreach ($snapshot in $snapshots) {
    $destination = Join-Path $root "assignments\$($snapshot.Year)\$($snapshot.Repo)"
    if ((Test-Path $destination) -and -not $Force) {
        Write-Host "Skip existing $($snapshot.Year)/$($snapshot.Repo)"
        continue
    }

    if (Test-Path $destination) {
        Remove-Item -Recurse -Force $destination
    }
    New-Item -ItemType Directory -Force -Path $destination | Out-Null

    $archive = Join-Path ([System.IO.Path]::GetTempPath()) "$($snapshot.Repo)-$($snapshot.Ref).tar.gz"
    $uri = "https://api.github.com/repos/stanford-cs336/$($snapshot.Repo)/tarball/$($snapshot.Ref)"
    Write-Host "Downloading $($snapshot.Year)/$($snapshot.Repo)"
    Invoke-WebRequest -UseBasicParsing -Uri $uri -Headers $headers -OutFile $archive -TimeoutSec 300
    & tar -xzf $archive -C $destination --strip-components=1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to extract $archive"
    }
    Remove-Item $archive -Force

    # Keep source, handouts, and test fixtures; omit bundled training corpora
    # and generated artifacts from this public learning repository.
    Get-ChildItem $destination -Directory -Recurse -Force |
        Where-Object { $_.Name -in @("data", "datasets", "checkpoints", "outputs", "runs", "wandb") } |
        Sort-Object FullName -Descending |
        ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
}

Write-Host "Official snapshots are up to date."
