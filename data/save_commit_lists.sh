#!/usr/bin/env bash

set -euo pipefail

script_dir="$(dirname $(realpath "$0"))"
pushd $script_dir >/dev/null

REPOS_FILE=github_repos.txt
CLONE_DIR=repos
COMMIT_LISTS_DIR=commit_lists

mkdir -p $COMMIT_LISTS_DIR

function has_exactly_one_parent() {
    full_repo_path=$1
    commit_hash=$2
    # Use the exit status of the last command in the pipeline as the "return value"
    # commits with multiple parents have spaces in the output.
    git -C $full_repo_path show --pretty=%p --quiet $commit_hash | grep -q '^[0-9a-f][0-9a-f]*$'
}

function save_commits_for_repo() {
    full_repo_path=$1

    echo "Saving commit list for repo '$full_repo_path'..."

    output_file=$COMMIT_LISTS_DIR/$(basename $full_repo_path).txt
    echo -n "" > $output_file
    for commit in $(git -C $full_repo_path rev-list HEAD); do
        if ! has_exactly_one_parent $full_repo_path $commit; then
            continue
        fi
        if ! has_exactly_one_parent $full_repo_path "$commit^"; then
            continue
        fi
        echo $commit >> $output_file
    done

    echo "Finished saving commit list for repo '$full_repo_path'"
}

while read -r git_repo_remote; do
    repo_dir=$(basename --suffix=.git $git_repo_remote)
    full_repo_path=$script_dir/$CLONE_DIR/$repo_dir
    save_commits_for_repo $full_repo_path &
done < <(grep -v '^#' $REPOS_FILE)

wait

popd
