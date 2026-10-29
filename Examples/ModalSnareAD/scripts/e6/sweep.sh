#!/bin/bash
# usage: sweep.sh NAME [latent-path options...]: fine-tune from base_e7, re-anchor, measure 100 walks.
# Runs in $MODAL_AD_ROOT/e6 (runs/ there); the shipped run is
#   sweep.sh F5 --lr 1e-2 --even 10 --len 0.3 --steps 4000 --readout-null --rec 0 --lip 0.05 --lip-cap 1.6 --pairs 16
set -e
CODE="$(cd "$(dirname "$0")" && pwd)"
BIN="${MODAL_AD_BIN:-$CODE/../../../../.build/release}"
cd "${MODAL_AD_ROOT:?set MODAL_AD_ROOT}/e6"
n=$1; shift
"$BIN/ModalSnareAD" latent-path --data runs/path_data_e7.json --init runs/base_e7/checkpoint.json --out runs/$n --save-every 100000 "$@" > runs/$n.log 2>&1
python3 "$CODE/reanchor.py" runs/base_e7 runs/$n runs/${n}_ra > runs/${n}_ra.log
python3 "$CODE/measure_walks.py" runs/${n}_ra runs/w100_${n}_ra.json --pairs 100 --workers 5 > /dev/null 2>&1
python3 - <<PY
import json; s=json.load(open('runs/w100_${n}_ra.json'))['summary']; b=s['band']; ra=json.load(open('runs/${n}_ra/reanchor.json'))
last=[l for l in open('runs/$n.log') if '|' in l][-1].split('|')[1].strip()
print('$n', '$*'.replace('--',''), '| dW3 %.2f | cv %.3f p90 %.3f maxmean %.2f first %.2f step %.2f'%(ra['dW3_rel'],b['cv_median'],b['cv_p90'],b['max_over_mean_median'],b['first_over_mean_median'],b['mean_step_median']), s['flags'], 'logf %.0f'%s['logf_dev_mid_cents_median'], '|', last)
PY
