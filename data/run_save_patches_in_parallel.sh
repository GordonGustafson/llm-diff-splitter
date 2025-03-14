#!/usr/bin/env bash

set -euo pipefail

script_dir="$(dirname $(realpath "$0"))"
pushd $script_dir >/dev/null

COMMIT_LISTS_DIR=commit_lists

combined_commit_lists_file=$(mktemp)

for file in $COMMIT_LISTS_DIR/*.txt; do
    sed "s/^/$(basename --suffix=.txt $file)_/" $file
done > $combined_commit_lists_file

cat $combined_commit_lists_file | parallel -m bash save_patches.sh

popd 
