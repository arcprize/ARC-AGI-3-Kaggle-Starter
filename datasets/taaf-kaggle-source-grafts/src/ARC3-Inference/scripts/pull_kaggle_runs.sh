#!/usr/bin/env bash
# Zieht fehlende Kaggle-Kernel-Ausgaben in die kanonischen Run-Ordner.
#
# Auf dem Mac ausführen (braucht kaggle-CLI + ~/.kaggle/kaggle.json):
#   cd ~/Desktop/duck-harness/ARC3-Inference
#   ./scripts/pull_kaggle_runs.sh              # nur fehlende/unvollständige
#   ./scripts/pull_kaggle_runs.sh --all        # alles neu ziehen
#   ./scripts/pull_kaggle_runs.sh 20260717_165455_stufe1b-k2   # gezielt
#
# Anders als beim alten Ablauf landet die Ausgabe NICHT in <run>/kaggle-output/,
# sondern direkt im Run-Ordner. Gründe:
#   1. `make view` findet Runs nur über den Ordnernamen (YYYYMMDD_HHMMSS_label)
#      und erwartet die Artefakte auf oberster Ebene.
#   2. inference/tools/eval.py nimmt den Verzeichnis-Basisnamen als Trial-Key —
#      mehrere Ordner namens "kaggle-output" kollidieren beim Poolen still.
# Die lokale Deploy-Variante von run_config.json bleibt als
# run_config.deploy.json erhalten; die Kaggle-Variante (mit hardware-Feld,
# das inference-significance braucht) gewinnt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/runs}"
FORCE=false
TARGETS=()

for arg in "$@"; do
  case "$arg" in
    --all|-a) FORCE=true ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) TARGETS+=("$arg") ;;
  esac
done

command -v kaggle >/dev/null || { echo "kaggle-CLI nicht gefunden (pip install kaggle)"; exit 1; }

cd "$RUNS_DIR"
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=($(ls -d 2*_* 2>/dev/null))

# Ein Run gilt als vollständig, wenn diese Dateien da sind.
required=(benchmark.json summary.txt stdout.log run_config.json)

pull() {
  local run="$1"
  local meta="$run/deploy_meta.json"
  if [ ! -f "$meta" ]; then
    echo "  $run: kein deploy_meta.json -> übersprungen"; return 0
  fi
  local kernel
  kernel="$(python3 -c "import json,sys; m=json.load(open('$meta')); print(m.get('job_id') or (m.get('target_config') or {}).get('kernel_id') or '')")"
  [ -n "$kernel" ] || { echo "  $run: kein kernel_id im deploy_meta -> übersprungen"; return 0; }

  local missing=()
  for f in "${required[@]}"; do [ -f "$run/$f" ] || missing+=("$f"); done
  if [ ${#missing[@]} -eq 0 ] && [ "$FORCE" != true ]; then
    echo "  $run: vollständig"; return 0
  fi
  echo "  $run: fehlt ${missing[*]:-<forced>}  <- $kernel"

  local status
  status="$(kaggle kernels status "$kernel" 2>&1 || true)"
  echo "    status: $(echo "$status" | tr '\n' ' ' | cut -c1-100)"
  case "$(echo "$status" | tr 'A-Z' 'a-z')" in
    *complete*|*error*|*cancel*) ;;
    *) echo "    -> läuft noch oder nicht abrufbar, übersprungen"; return 0 ;;
  esac

  local tmp="$run/.pull-tmp"
  rm -rf "$tmp"; mkdir -p "$tmp"
  if ! kaggle kernels output "$kernel" -p "$tmp"; then
    echo "    -> Download fehlgeschlagen"; rm -rf "$tmp"; return 0
  fi
  if [ ! -f "$tmp/benchmark.json" ]; then
    echo "    -> WARNUNG: keine benchmark.json im Output (Submission-Kernel schreiben keine)"
  fi

  # Kaggle-Ausgabe eine Ebene hoch mergen
  [ -f "$run/run_config.json" ] && [ -f "$tmp/run_config.json" ] \
    && mv "$run/run_config.json" "$run/run_config.deploy.json"
  for item in "$tmp"/* "$tmp"/.[!.]*; do
    [ -e "$item" ] || continue
    base="$(basename "$item")"
    rm -rf "${run:?}/$base"
    mv "$item" "$run/"
  done
  rmdir "$tmp"
  echo "    -> ok ($(ls "$run" | wc -l | tr -d ' ') Einträge, $(du -sh "$run" | cut -f1))"
}

echo "runs dir: $RUNS_DIR"
for run in "${TARGETS[@]}"; do
  run="${run%/}"
  [ -d "$run" ] || { echo "  $run: nicht gefunden"; continue; }
  [ -L "$run" ] && continue
  pull "$run"
done

echo
echo "Stand:"
for d in 2*_*; do
  [ -L "$d" ] && continue
  printf '  %-46s %s\n' "$d" "$([ -f "$d/benchmark.json" ] && echo "$(python3 -c "import json;print(len(json.load(open('$d/benchmark.json'))['game_runs']),'game runs')" 2>/dev/null)" || echo 'LEER')"
done
echo
echo "Danach:  make view VIEW_RUN_DIR= VIEW_RUNS_DIR=runs"
