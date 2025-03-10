#!/usr/bin/env bash

set -euo pipefail

REPOS_FILE=github_repos.txt
CLONE_DIR=repos
OUTPUT_PATCHFILE_DIR=patches

mkdir -p $OUTPUT_PATCHFILE_DIR

function is_merge_commit() {
    full_repo_path=$1
    commit_hash=$2
    parents=$(git -C $full_repo_path show --pretty=%ph --quiet $commit_hash)
    # It's a merge commit if there's multiple parents (if the list of parents has a space in it).
    if echo "$parents" | grep -q " "; then
        true
    else
        false
    fi
}

while read git_repo_remote; do
    repo_dir=$(basename --suffix=.git $git_repo_remote)
    full_repo_path=$CLONE_DIR/$repo_dir
    echo "Saving patches for repo '$repo_dir'..."
    # Skip first two commits in the repo because we need commits to have two parents.
    for commit in $(git -C $full_repo_path rev-list HEAD | head -n -2); do
        if is_merge_commit $full_repo_path $commit; then
            continue
        fi
        if is_merge_commit $full_repo_path $commit^; then
            continue
        fi
        output_filename="$OUTPUT_PATCHFILE_DIR/$repo_dir-$commit.tar"
        git -C $full_repo_path diff $commit^^..$commit^ > first-diff.patch
        git -C $full_repo_path diff $commit^..$commit > second-diff.patch
        git -C $full_repo_path diff $commit^^..$commit > combined-diff.patch
        tar cf $output_filename first-diff.patch second-diff.patch combined-diff.patch
    done
done < $REPOS_FILE
