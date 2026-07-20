#!/usr/bin/env bash
set -euo pipefail

# v0.7.1 keeps the v0.6 script path as a compatibility alias for existing
# automation, while this is the release-facing entrypoint.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/regression_v06.sh" "$@"
