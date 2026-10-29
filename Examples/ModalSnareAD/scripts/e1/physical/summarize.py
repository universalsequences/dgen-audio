import json, glob, os
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e1/physical")
for d in sorted(glob.glob(os.path.join(HERE, "*", "phys_summary.json"))):
    s = json.load(open(d)); name = os.path.basename(os.path.dirname(d))
    fit = s["per_snare_fit"]; wr = s["wrong_min_head"]
    beat = sum(f < w / 1.5 for f, w in zip(fit, wr))
    print(f"{name:12s} lambda={s['lambda']:<5} init {s['mean_init']:.4f} fit {s['mean_fit']:.4f} wrong-min {s['mean_wrong_min_head']:.4f} "
          f"fit<wrong/1.5: {beat}/32  res f {s['res_f_cents_rms']:.0f} c rms, dec {s['res_dec_log_rms']:.2f}, amp {s['res_amp_log_rms']:.2f} (log rms)  {s['s_per_step_median']:.2f} s/step")
