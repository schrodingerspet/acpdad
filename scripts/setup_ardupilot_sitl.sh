#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARDUPILOT_HOME="${ARDUPILOT_HOME:-${PROJECT_ROOT}/.sitl/ardupilot}"
ARDUPILOT_TAG="Copter-4.7.2"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"

if ! command -v git >/dev/null 2>&1; then
  echo "Missing prerequisite: git. Install git, then rerun this script." >&2
  exit 2
fi

if [[ ! -d "${ARDUPILOT_HOME}/.git" ]]; then
  if [[ -e "${ARDUPILOT_HOME}" ]]; then
    echo "ARDUPILOT_HOME exists but is not an ArduPilot Git checkout: ${ARDUPILOT_HOME}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${ARDUPILOT_HOME}")"
  git clone --branch "${ARDUPILOT_TAG}" --depth 1 --recurse-submodules \
    https://github.com/ArduPilot/ardupilot.git "${ARDUPILOT_HOME}"
fi

actual_tag="$(git -C "${ARDUPILOT_HOME}" tag --points-at HEAD 2>/dev/null | grep -Fx "${ARDUPILOT_TAG}" || git -C "${ARDUPILOT_HOME}" describe --tags --exact-match HEAD 2>/dev/null || true)"
if [[ "${actual_tag}" != "${ARDUPILOT_TAG}" ]]; then
  echo "Expected ArduPilot tag ${ARDUPILOT_TAG}, found '${actual_tag:-untagged HEAD}' at ${ARDUPILOT_HOME}." >&2
  echo "Use a clean checkout at the pinned tag; the script will not switch an existing checkout." >&2
  exit 2
fi

if [[ -d "${PROJECT_ROOT}/.venv/bin" ]]; then
  export PATH="${PROJECT_ROOT}/.venv/bin:${PATH}"
fi

if [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]]; then
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
  fi
  case "${ID:-}" in
    arch)
      prereqs="${ARDUPILOT_HOME}/Tools/environment_install/install-prereqs-arch.sh"
      prereq_args=(-qy)
      ;;
    ubuntu|debian)
      prereqs="${ARDUPILOT_HOME}/Tools/environment_install/install-prereqs-ubuntu.sh"
      prereq_args=(-y)
      ;;
    *)
      echo "Unsupported Linux distribution '${ID:-unknown}' for the bundled prerequisite helper." >&2
      echo "Install ArduPilot's documented build prerequisites manually, then rerun with INSTALL_SYSTEM_DEPS=0." >&2
      exit 2
      ;;
  esac
  if [[ ! -f "${prereqs}" ]]; then
    echo "ArduPilot prerequisite script is missing: ${prereqs}" >&2
    exit 2
  fi
  bash "${prereqs}" "${prereq_args[@]}"
fi

cd "${ARDUPILOT_HOME}"
./waf configure --board sitl
./waf copter

echo "ArduPilot SITL build is ready."
echo "ARDUPILOT_HOME=${ARDUPILOT_HOME}"
echo "ARDUPILOT_TAG=${actual_tag}"
git -C "${ARDUPILOT_HOME}" rev-parse HEAD
