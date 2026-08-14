# Maintainer-only: publish SightLoom crates to crates.io.
#
# The API token is a *secret*. Never commit it.
#
#   1. Create token: https://crates.io/settings/tokens
#   2. Set environment variable for this shell only:
#        $env:CARGO_REGISTRY_TOKEN = 'cio_...'
#   3. From repo root:
#        ./scripts/publish-crates.ps1
#        ./scripts/publish-crates.ps1 -DryRun
#
# CI: store the same value as GitHub Actions secret CARGO_REGISTRY_TOKEN
# (see .github/workflows/publish.yml). Do not document the token value anywhere.

param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not $env:CARGO_REGISTRY_TOKEN) {
    Write-Error @"
Environment variable CARGO_REGISTRY_TOKEN is not set (publish secret).

Create a token at https://crates.io/settings/tokens
Then in this shell only:
  `$env:CARGO_REGISTRY_TOKEN = 'cio_...'

Do not put the token in README or git.
"@
}

$order = @(
    "sightloom-core",
    "sightloom-tracking",
    "sightloom-analysis",
    "sightloom-reid",
    "sightloom-index",
    "sightloom",
    "sightloom-host"
)

foreach ($crate in $order) {
    Write-Host "==== $crate ====" -ForegroundColor Cyan
    $cargoArgs = @("publish", "-p", $crate, "--locked")
    if ($DryRun) {
        $cargoArgs += "--dry-run"
    }
    & cargo @cargoArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "cargo publish failed for $crate (exit $LASTEXITCODE)"
    }
    if (-not $DryRun -and $crate -ne "sightloom") {
        Start-Sleep -Seconds 30
    }
}

Write-Host "Done." -ForegroundColor Green
