#!/bin/bash
set -e

run_towncrier_check () {
  echo "BITBUCKET_TAG is not set or does not start with 'v'. Running fragment check..."
  pip install -U pip
  pip install towncrier
  git remote set-branches --add origin main
  git fetch origin main
  towncrier check --compare-with origin/main
}

check_CHANGELOG_updated () {
  echo "Checking that CHANGELOG has been built for the current release (${BITBUCKET_TAG})..."
  local ERROR_PAD="***********"

  if [[ $BITBUCKET_TAG =~ "rc" ]]; then
    echo "Release candidate detected. No check needed."
    exit 0;
  fi

  ls changelog/*rst 2>/dev/null && {
    echo "${ERROR_PAD}"
    echo -e "Unmerged changelog fragments found! You must run towncrier!\\nSee the README for instructions"
    echo "${ERROR_PAD}"
    exit 1;
  }

  head -n 1 CHANGELOG.rst | grep -q "^$BITBUCKET_TAG (" - || {
    echo "${ERROR_PAD}"
    echo -e "CHANGELOG has not been updated for ${BITBUCKET_TAG}\\nSee the README for instructions"
    echo "${ERROR_PAD}"
    exit 1;
  }
  echo "Success"
}

if [[ -n "$BITBUCKET_TAG" && "$BITBUCKET_TAG" == v* ]]; then
  check_CHANGELOG_updated
else
  run_towncrier_check
fi
