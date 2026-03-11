#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

GCP_PROJECT="${PROJECT:-cerebra-shodan-ci-public}"
GCP_LOCATION="${LOCATION:-us}"
AR_REPO="${REPO:-coralnpu-artifacts}"
AR_PACKAGE="${PACKAGE:-coralnpu-bitstreams}"
TARGET_FILE="${TARGET:-chip_nexus.bin}"
LIMIT=10
START_REF=""

usage() {
    echo "Usage:"
    echo "  get_bitsream.sh (--latest|--ref REF) [options]"
    echo ""
    echo "Options:"
    echo "  -r, --ref      The git reference to start searching from"
    echo "  --latest       Search starting from the current HEAD"
    echo "  -n, --limit    Number of git commits to check back (default: ${LIMIT})"
    echo "  --target       The name of the file to fetch (default: ${TARGET_FILE})"
    echo "  -h, --help     Display this help message and exit"
    echo ""
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case "${1}" in
        -n|--limit)
            LIMIT="$2"; shift 2 ;;
        --target)
            TARGET_FILE="$2"; shift 2 ;;
        -r|--ref)
            START_REF="$2"; shift 2 ;;
        --latest)
            START_REF="HEAD"; shift 1 ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "${START_REF}" ]]; then
    echo "Error: Either --ref or --latest must be specified."
    echo ""
    usage
fi

# Locate project root relative to script location
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Verify Git availability in project root
git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree > /dev/null 2>&1
if [[ $? -ne 0 ]]; then
    echo "Error: git not found or git history not available in ${PROJECT_ROOT}."
    exit 1
fi

echo "Searching for the most recent artifact in Artifact Registry..."
echo "Start Ref:    ${START_REF}"
echo "Search limit: ${LIMIT} commits"
echo "Target File:  ${TARGET_FILE}"
echo ""

LAST_SHA=""
for (( i=0; i<$LIMIT; i++ )); do
    # Switch to project root to resolve Git SHAs
    pushd "${PROJECT_ROOT}" > /dev/null || exit 1
    SHA=$(git rev-parse "${START_REF}~$i" 2>/dev/null)
    popd > /dev/null || exit 1

    if [[ -z "${SHA}" ]]; then
        echo "Reached end of local git history or invalid reference."
        break
    fi

    LAST_SHA="${SHA}"

    # Artifact Registry Generic format uses colons as delimiters.
    # In the HTTP REST API, these colons MUST be URL-encoded as %3A.
    URL="https://artifactregistry.googleapis.com/download/v1"
    URL+="/projects/${GCP_PROJECT}"
    URL+="/locations/${GCP_LOCATION}"
    URL+="/repositories/${AR_REPO}"
    URL+="/files/${AR_PACKAGE}%3A${SHA}%3A${TARGET_FILE}:download?alt=media"

    echo "[$i] Checking commit ${SHA:0:8}..."

    if curl -sL --fail -o "${TARGET_FILE}" "${URL}"; then
        echo "SUCCESS: Found artifact for commit ${SHA}"
        echo "File saved as: ${TARGET_FILE}"
        exit 0
    fi
done

echo "ERROR: No matching artifact found in the previous ${LIMIT} commits."
echo "The artifact might not have been built for these specific commits, or the build is still in progress."
echo ""
echo "To search further back, try increasing --limit or running:"
echo "  ${0} --ref ${LAST_SHA}^ --target ${TARGET_FILE}"
exit 1
