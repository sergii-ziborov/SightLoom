#!/usr/bin/env bash
# Maintainer-only: publish SightLoom crates to crates.io.
# Token is a secret — never commit it or put it in README.
#
#   export CARGO_REGISTRY_TOKEN=cio_...   # this shell only
#   ./scripts/publish-crates.sh
#   ./scripts/publish-crates.sh --dry-run

set -euo pipefail

if [[ -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
  cat <<'EOF' >&2
CARGO_REGISTRY_TOKEN is not set (publish secret).

Create a token: https://crates.io/settings/tokens
Then (this shell only):
  export CARGO_REGISTRY_TOKEN=cio_...

Do not put the token in README or git.
EOF
  exit 1
fi

DRY=()
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=(--dry-run)
fi

ORDER=(
  sightloom-core
  sightloom-tracking
  sightloom-analysis
  sightloom-reid
  sightloom-index
  sightloom
)

for crate in "${ORDER[@]}"; do
  echo "==== ${crate} ===="
  cargo publish -p "${crate}" --locked "${DRY[@]}"
  if [[ ${#DRY[@]} -eq 0 && "${crate}" != "sightloom" ]]; then
    sleep 30
  fi
done

echo "Done."
