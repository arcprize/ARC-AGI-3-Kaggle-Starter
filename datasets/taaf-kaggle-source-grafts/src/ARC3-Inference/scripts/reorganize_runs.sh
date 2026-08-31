#!/usr/bin/env bash
# Einmalige Aufräumaktion für ARC3-Inference/runs (2026-07-27).
#
# Ziel: jeder echte Run liegt als kanonischer Ordner YYYYMMDD_HHMMSS_<label>
# DIREKT unter runs/, mit den Artefakten auf oberster Ebene. Nur so findet
# `make view` (viewer/data.py -> is_selectable_run_dir_name) die Runs
# automatisch, und nur so ist der Trial-Key-Bug in inference/tools/eval.py
# umgangen (Basename "kaggle-output" kollidiert sonst über alle Runs).
#
# Es wird nichts gelöscht: alles Überflüssige landet unter runs/_archive/.
# Idempotent: bereits erledigte Schritte werden übersprungen.
set -euo pipefail

RUNS_DIR="${RUNS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/runs}"
cd "$RUNS_DIR"
echo "runs dir: $RUNS_DIR"
echo

mkdir -p _archive/leftover-dirs _archive/superseded-files _analysis

# stash <pfad> <praefix> — verschiebt statt zu löschen
stash() {
  local src="$1" tag="$2" dest
  [ -e "$src" ] || return 0
  dest="_archive/${3:-superseded-files}/${tag}"
  mkdir -p "$(dirname "$dest")"
  mv "$src" "$dest"
}

# --- 1. macOS-Müll -------------------------------------------------------
echo "[1] .DS_Store einsammeln"
i=0
while IFS= read -r f; do
  i=$((i+1)); mv "$f" "_archive/superseded-files/DS_Store_$i" 2>/dev/null || true
done < <(find . -name '.DS_Store' -not -path './_archive/*' 2>/dev/null)
[ -f .per_game_cards.pkl ] && mv .per_game_cards.pkl _archive/superseded-files/per_game_cards-runs-root.pkl
echo "  $i .DS_Store archiviert"
echo

# --- 2. kaggle-output/ eine Ebene hochziehen -----------------------------
# Die auf Kaggle geschriebene run_config.json gewinnt (enthält das
# hardware-Feld, das inference-significance für die Kompatibilitätschecks
# braucht). Die lokale Deploy-Variante bleibt als run_config.deploy.json.
echo "[2] kaggle-output hochziehen"
promote() {
  local run="$1"
  [ -d "$run/kaggle-output" ] || return 0
  echo "  $run/kaggle-output -> $run/"
  [ -f "$run/run_config.json" ] && mv "$run/run_config.json" "$run/run_config.deploy.json"
  for item in "$run/kaggle-output"/* "$run/kaggle-output"/.[!.]*; do
    [ -e "$item" ] || continue
    local base; base="$(basename "$item")"
    if [ -e "$run/$base" ]; then stash "$run/$base" "${run}__${base}"; fi
    mv "$item" "$run/"
  done
  mv "$run/kaggle-output" "_archive/leftover-dirs/${run}__kaggle-output"
}
for run in 20260720_102119_duck-noop-batch-20260720-on \
           20260720_102123_duck-noop-batch-20260720-off \
           20260720_141639_duck-noop-batch-20260720-on-2 \
           20260720_141646_duck-noop-batch-20260720-off-2; do
  promote "$run"
done
echo

# --- 3. k1k2 in seine Deploy-Hülle mergen --------------------------------
# runs/k1k2 ist die manuell gezogene Ausgabe des Kernels taaf-duckk1k2.
# Der Kernel-Lauf startete 2026-07-14 10:25 CEST, also aus dem dritten
# Deploy-Versuch (10:15:26). Die beiden früheren Hüllen sind redundant
# (identischer git_status, gleicher Kernel, kein eigener Output).
echo "[3] k1k2 mergen"
if [ -d k1k2 ] && [ ! -L k1k2 ]; then
  for item in k1k2/* k1k2/.[!.]*; do
    [ -e "$item" ] || continue
    base="$(basename "$item")"
    [ -e "20260714_101526_duckk1k2/$base" ] && stash "20260714_101526_duckk1k2/$base" "duckk1k2__${base}"
    mv "$item" 20260714_101526_duckk1k2/
  done
  mv k1k2 _archive/leftover-dirs/k1k2-empty
  ln -s 20260714_101526_duckk1k2 k1k2
  echo "  k1k2 -> 20260714_101526_duckk1k2 (+ Kompat-Symlink)"
fi
for stale in 20260714_092336_duckk1k2 20260714_095220_duckk1k2; do
  if [ -d "$stale" ] && [ ! -L "$stale" ]; then
    mv "$stale" _archive/ && echo "  archiviert (redundanter Deploy): $stale"
  fi
done
echo

# --- 4. Nicht-kanonisch benannte Runs umbenennen -------------------------
# Zeitstempel = deploy_meta.json started_at (lokale Mac-Zeit, gleiche
# Konvention wie die vom Harness selbst erzeugten Ordnernamen).
echo "[4] kanonische Namen"
rename_run() {
  local old="$1" new="$2"
  [ -d "$old" ] && [ ! -L "$old" ] || return 0
  mv "$old" "$new"
  ln -s "$new" "$old"          # Rückwärtskompatibilität für alte Pfade/Notizen
  echo "  $old -> $new (+ Kompat-Symlink)"
}
rename_run stufe1-k2          20260712_100911_stufe1-k2
rename_run stufe1-baseline    20260712_101024_stufe1-baseline
rename_run duck-k2-submission 20260713_095003_duck-k2-submission
rename_run stufe1b-k2         20260717_165455_stufe1b-k2
rename_run stufe1b-baseline   20260717_165703_stufe1b-baseline
echo

# --- 5. Analyse-Output aus runs/ raus ------------------------------------
echo "[5] Analyse-Output"
if [ -d pooled ] && [ ! -L pooled ]; then
  mv pooled _analysis/ && ln -s _analysis/pooled pooled && echo "  pooled -> _analysis/pooled"
fi
echo

echo "Kanonische Runs:"
for d in 2*_*; do [ -L "$d" ] && continue; printf '  %-46s %s\n' "$d" "$([ -f "$d/benchmark.json" ] && echo 'Daten vorhanden' || echo 'LEER - Kaggle-Pull ausstehend')"; done
