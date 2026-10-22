#!/usr/bin/env bash
# User training recipe. Run inside a scheduled K100 allocation.
# Usage: bash train_hco.sh short|q|qmu|qmuq DATA_DIR CHECKPOINT [extra mff-train arguments]
set -euo pipefail
if [[ $# -lt 3 ]]; then
    printf 'Usage: bash train_hco.sh short|q|qmu|qmuq DATA_DIR CHECKPOINT [extra arguments]\n' >&2
    exit 2
fi
mode="$1"; data_dir="$2"; checkpoint="$3"
shift 3
e0_csv="${E0_CSV:-$data_dir/fitted_E0.csv}"
mapfile -t e0 < <(python - "$e0_csv" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise ValueError("The E0 CSV is empty")
print(",".join(str(int(row["Atom"])) for row in rows))
print(",".join(str(float(row["E0"])) for row in rows))
PY
)
[[ ${#e0[@]} -eq 2 ]] || { printf 'Could not read Atom,E0 from %s\n' "$e0_csv" >&2; exit 2; }
args=(
    --data-dir "$data_dir" --checkpoint "$checkpoint"
    --atomic-energy-keys="${e0[0]}" --atomic-energy-values="${e0[1]}"
    --channels 64 --lmax 2 --max-ell 2 --num-interaction 2 --correlation 3
    --max-radius 4.5 --polynomial-cutoff-p 6
    --product-backend ictd-bridge-u --angular-basis ictd --freeze-conv-tp-weight
    --device cuda --dtype float32 --operator-plan generated --generated-layout fused_pack
    --training-message-linear-layout node_component
    --optimizer adamw --lr 1e-3 --loss mse --energy-weight 1 --force-weight 10
    --batch-size 4 --epochs 100 --num-workers 2 --seed 42
    --long-range-dispersion-mode none
)
case "$mode" in
    short) args+=(--long-range-mode none) ;;
    q|qmu|qmuq)
        case "$mode" in q) rank=0 ;; qmu) rank=1 ;; qmuq) rank=2 ;; esac
        args+=(
            --long-range-mode reciprocal-spectral-v1 --long-range-boundary periodic
            --long-range-reciprocal-backend mesh_fft
            --long-range-mesh-size "${ICTC_MESH:-32,32,32}"
            --long-range-source-channels 1 --long-range-max-multipole-l "$rank"
            --long-range-assignment pcs --long-range-green-mode poisson
            --long-range-mesh-fft-full-ewald --long-range-mesh-fft-reciprocal-only
            --long-range-smearing-sigma 1.0 --electrostatics-backend hip
        ) ;;
    *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$checkpoint")"
exec mff-train "${args[@]}" "$@"
