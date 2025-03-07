#!/usr/bin/env bash

set -euo pipefail

REPOS_FILE=github_repos.txt
CLONE_DIR=repos
OUTPUT_PATCHFILE_DIR=patches
DIFF_FILE_SPLITTER=c754a7f4bacb45eda4ed3f2b9ec2fdb7

mkdir -p $OUTPUT_PATCHFILE_DIR

while read git_repo_remote; do
    repo_dir=$(basename --suffix=.git $git_repo_remote)
    full_repo_path=$CLONE_DIR/$repo_dir
    echo "Saving patches for repo '$repo_dir'..."
    # Skip first two commits in the repo because we need commits to have two parents.
    for commit in $(git -C $full_repo_path rev-list HEAD | head -n -2); do
        output_filename="$OUTPUT_PATCHFILE_DIR/$repo_dir-$commit.multipatch"
        git -C $full_repo_path diff $commit^..$commit > $output_filename
        echo $DIFF_FILE_SPLITTER >> $output_filename
        git -C $full_repo_path diff $commit^^..$commit^ >> $output_filename
        echo $DIFF_FILE_SPLITTER >> $output_filename
        git -C $full_repo_path diff $commit^^..$commit >> $output_filename
    done
done < $REPOS_FILE
