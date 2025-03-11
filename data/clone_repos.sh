#!/usr/bin/env bash

set -euo pipefail

pushd "$(dirname "$0")"

REPOS_FILE=github_repos.txt
CLONE_DIR=repos

mkdir -p $CLONE_DIR

while read git_repo_remote; do
    pushd $CLONE_DIR
    git clone $git_repo_remote
    popd
done < $REPOS_FILE
popd
