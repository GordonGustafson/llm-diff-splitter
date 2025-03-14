#!/usr/bin/env bash

set -euo pipefail

script_dir="$(dirname $(realpath "$0"))"
pushd $script_dir >/dev/null

CLONE_DIR=repos
OUTPUT_PATCHFILE_DIR=$script_dir/patches

mkdir -p $OUTPUT_PATCHFILE_DIR

FIRST_PATCH_FILENAME=first-diff.patch
SECOND_PATCH_FILENAME=second-diff.patch
COMBINED_PATCH_FILENAME=combined-diff.patch


tempdir=$(mktemp -d)
pushd $tempdir >/dev/null
for pair in "$@"; do
    # $pair is of the form $repo_$commit, so we split them apart.
    repo="${pair%_*}"
    commit="${pair#*_}"

    full_repo_path=$script_dir/$CLONE_DIR/$repo
    output_filename="$OUTPUT_PATCHFILE_DIR/$repo-$commit.tar"
    git -C $full_repo_path diff $commit^^..$commit^ > $FIRST_PATCH_FILENAME &
    git -C $full_repo_path diff $commit^..$commit > $SECOND_PATCH_FILENAME &
    git -C $full_repo_path diff $commit^^..$commit > $COMBINED_PATCH_FILENAME &
    wait
    tar cf $output_filename $FIRST_PATCH_FILENAME $SECOND_PATCH_FILENAME $COMBINED_PATCH_FILENAME
done
popd
rm -rf $tempdir

popd
