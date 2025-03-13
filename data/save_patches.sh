#!/usr/bin/env bash

set -euo pipefail

script_dir="$(dirname $(realpath "$0"))"
pushd $script_dir >/dev/null

echo $script_dir

REPOS_FILE=github_repos.txt
CLONE_DIR=repos
OUTPUT_PATCHFILE_DIR=$script_dir/patches

mkdir -p $OUTPUT_PATCHFILE_DIR

FIRST_PATCH_FILENAME=first-diff.patch
SECOND_PATCH_FILENAME=second-diff.patch
COMBINED_PATCH_FILENAME=combined-diff.patch

function has_exactly_one_parent() {
    full_repo_path=$1
    commit_hash=$2
    parents=$(git -C $full_repo_path show --pretty=%p --quiet $commit_hash)
    # If the list of parents has a space in it, return false.
    if echo "$parents" | grep -q " "; then
        return 1
    fi
    # If there's something in the list of parents, return true.
    if echo "$parents" | grep -q "."; then
        return 0
    fi
    return 1
}

function save_commits_for_repo() {
    full_repo_path=$1

    echo "Saving patches for repo '$full_repo_path'..."

    tempdir=$(mktemp -d)
    pushd $tempdir >/dev/null
    for commit in $(git -C $full_repo_path rev-list HEAD); do
        if ! has_exactly_one_parent $full_repo_path $commit; then
            continue
        fi
        if ! has_exactly_one_parent $full_repo_path "$commit^"; then
            continue
        fi
        output_filename="$OUTPUT_PATCHFILE_DIR/$repo_dir-$commit.tar"
        git -C $full_repo_path diff $commit^^..$commit^ > $FIRST_PATCH_FILENAME
        git -C $full_repo_path diff $commit^..$commit > $SECOND_PATCH_FILENAME
        git -C $full_repo_path diff $commit^^..$commit > $COMBINED_PATCH_FILENAME
        tar cf $output_filename $FIRST_PATCH_FILENAME $SECOND_PATCH_FILENAME $COMBINED_PATCH_FILENAME
    done
    popd
    rm -rf $tempdir

    echo "Finished saving patches for repo '$full_repo_path'"
}

# Taken from https://stackoverflow.com/a/2873039 . My guess is that this would
# only speed up `git diff` with the current index rather than `git diff` from
# two previous commits, but I'm giving it a try.
git config core.preloadindex true

while read -r git_repo_remote; do
    repo_dir=$(basename --suffix=.git $git_repo_remote)
    full_repo_path=$script_dir/$CLONE_DIR/$repo_dir
    save_commits_for_repo $full_repo_path  &
done < $REPOS_FILE

wait

popd
