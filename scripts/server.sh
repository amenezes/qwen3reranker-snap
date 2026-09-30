#!/bin/bash

set -euo pipefail

# --wait-for-components waits for the weight/runtime components to be mounted
# before the server starts: on a fresh install the daemon can come up while the
# components are still being installed, and then MODEL_FILE does not exist yet.
# "engine" replaces the deprecated "show-engine" (see `modelctl show-engine`).
engine="$(modelctl status --wait-for-components --format=json | jq -r .engine)"
exec modelctl run --share-provider -- "$SNAP/engines/$engine/server" "$@"
