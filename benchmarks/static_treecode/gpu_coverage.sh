#!/usr/bin/env bash
# Usage: gpu_coverage.sh [--dry-run]
#
# Coverage of the GPU tests, measured on an A100 and uploaded to Codecov as the
# "gpu" flag. CI uploads the CPU tests as "cpu"; Codecov merges the two per commit
# and carries the last gpu report forward to commits without one (codecov.yml).
# Run it after changing GPU code and before a release.
#
# Steps: deploy this tree to el9 (deploy_el9.sh), run `pytest -m gpu` with coverage on
# an A100 node (remote_run.sh), upload the XML from el9-login for the current HEAD.
# The upload is attributed to HEAD, so the tree must be committed and HEAD pushed.
# A failing GPU test stops the script before the upload.
#
# One-time setup on el9 (done 2026-11-03):
#   ~/.config/codecov/pixpoint-token  the repository upload token (mode 600)
#   ~/.local/bin/codecov              Codecov's uploader (codecovcli), checked against
#                                     its SHA256SUM and signature
set -euo pipefail
cd "$(dirname "$0")/../.."
REMOTE=/project/ocean1-seitz/users/dgebauer/research/CosmoFuse-static-treecode
DRY_RUN=""
[ "${1:-}" = "--dry-run" ] && DRY_RUN="--dry-run"

# The deploy copies the working tree, the upload names a commit: they must agree.
if [ -n "$(git status --porcelain --untracked-files=no)" ] ||
   [ -n "$(git status --porcelain -- src tests)" ]; then
    echo "gpu_coverage.sh: uncommitted changes (or untracked files in src/ or tests/); commit first" >&2
    exit 1
fi
sha=$(git rev-parse HEAD)
branch=$(git rev-parse --abbrev-ref HEAD)
git fetch -q origin
if [ -z "$(git branch -r --contains "$sha")" ]; then
    echo "gpu_coverage.sh: HEAD $sha is not on GitHub; push first" >&2
    exit 1
fi

benchmarks/static_treecode/deploy_el9.sh
benchmarks/static_treecode/remote_run.sh \
    "rm -f coverage-gpu.xml .coverage && pytest -m gpu -q -p no:cacheprovider --cov=pixpoint --cov-report=xml:coverage-gpu.xml"

ssh -o BatchMode=yes el9-login "cd $REMOTE && CODECOV_TOKEN=\$(cat ~/.config/codecov/pixpoint-token) \
    ~/.local/bin/codecov upload-process $DRY_RUN --fail-on-error --git-service github \
    --slug D-Gebauer/PixPoint --sha $sha --branch $branch --flag gpu --name a100 \
    --file coverage-gpu.xml --disable-search --network-root-folder $REMOTE"
echo "gpu coverage of $sha uploaded: https://app.codecov.io/gh/D-Gebauer/PixPoint/commit/$sha"
