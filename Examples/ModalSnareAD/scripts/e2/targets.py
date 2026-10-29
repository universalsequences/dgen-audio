#!/usr/bin/env python3
"""Teacher targets for E2 supervised pretraining, in the realtime template's
NATURAL parameter space (one dict of arrays per render).

Head (K=32, VILLAIN's own slots): villain_modes.json, mapped to the template:
  f = table f * tuning(C4) (heard Hz); amp = table amp * head_v0 * gin;
  decay = dec0; glide g = max(-0.9, g0); phase wrapped to (-pi, pi].
Gates: VILLAIN GH/GD with all-mode masks and no GD sag (gate_impact2.py: MR 0.0019).
Noise + output stage: E1 fits (e1/fits/params.json, v100). Lower velocities reuse the
snare's v100 noise with A scaled by the (source-head) stem RMS ratio, and gain by the
table's lvl*vel^c_vout ratio (E1's fitted/table gain ratio kept).
    python3 targets.py  -> targets.json
"""
import json, math, os, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e2"); R = MAD.ROOT
MAD.use("dataset")
from villain_common import read_wav_f32
TUN = 2 * 261.6255653 / 440
VELS = {'040': 0.4, '070': 0.7, '100': 1.0}
d = json.load(open(os.path.join(R, 'data/villain_modes.json')))['renders']
e1 = {n['id']: n for n in json.load(open(os.path.join(R, 'e1/fits/params.json')))['natural']}
man = {i['id']: i for i in json.load(open(os.path.join(R, 'data/villain/manifest.json')))['items']}

def vp(v, e): return math.exp(e * math.log(max(v, 1e-4))) if v > 0 else 0.0

def rms(x): return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))

def noise_rms(rid):
    s = read_wav_f32(os.path.join(R, f'data/villain/stems/{rid}.source.wav'))
    h = read_wav_f32(os.path.join(R, f'data/villain/stems/{rid}.head.wav'))
    return rms(s - h)

def target(snare, vtag):
    rid = f'snare_{snare:02d}_v{vtag}'; vel = VELS[vtag]
    fit = dict(e1[f'snare_{snare:02d}_v100'])
    fit['_nscale'] = 1.0 if vtag == '100' else noise_rms(rid) / max(noise_rms(f'snare_{snare:02d}_v100'), 1e-9)
    return rid, vel, target_from(d[rid], vel, fit)

def target_from(r, vel, fit):
    """r: a villain_modes-style render {slots, chain, global}; fit: E1 natural params
    (noise + output stage at v1.0) plus '_nscale' (noise level at this velocity)."""
    c = r['chain']; g = r['global']
    gin = vp(vel, -c['c_vin']); hv = c['c_head'] * vp(vel, c['c_hv'])
    S = r['slots']
    # pure table head (E1's out-fitted head residuals would fight the head-stem stage)
    fm = am = dm = np.ones(32)
    t = {}
    t['mode_freq'] = np.array([s['f'] for s in S]) * TUN * fm
    t['mode_amp'] = np.array([s['amp'] for s in S]) * hv * gin * am
    t['mode_decay'] = np.array([s['dec0'] for s in S]) / dm
    t['mode_rise'] = np.array([s['rise0'] for s in S])
    t['mode_t0'] = np.array([s['t0'] for s in S])
    t['mode_phase'] = np.angle(np.exp(1j * np.array([s['ph'] for s in S])))
    t['mode_glide'] = np.maximum(-0.9, np.array([s['g0'] for s in S]))
    t['mode_gtau'] = np.array([s['tau'] for s in S])
    # noise: E1 v100 fit, A scaled to this velocity
    nscale = fit['_nscale']
    gin100 = vp(1.0, -c['c_vin'])
    t['band_amp'] = np.array(fit['noise_amp']) * gin100 * nscale
    t['band_d1'] = np.array(fit['noise_d1_s']); t['band_tail'] = np.array(fit['noise_tail_frac'])
    t['band_d2'] = np.array(fit['noise_d2_s']); t['band_couple'] = np.array(fit['noise_couple'])
    gl = {}
    gl['noise_rise'] = fit['noise_rise_s']; gl['noise_onset'] = fit['noise_onset_s']; gl['pre'] = c['c_pre']
    gl['drive'] = fit['drive']; gl['drive_amt'] = fit['drive_amt']; gl['drive_norm'] = g['d_norm']
    tab_gain = lambda v: g['lvl'] * vp(v, c['c_vout'])
    gl['gain'] = fit['out_gain'] * (tab_gain(vel) / max(tab_gain(1.0), 1e-9))
    gl['len'] = g['len_s']; gl['fade'] = max(g['fade_s'], 1e-4); gl['fade_curve'] = c['c_fade_curve']
    gl['gh_depth'] = c['c_gh_depth']; gl['gh_time'] = c['c_gh_ms']; gl['gh_fall'] = c['c_gh_fall']
    gl['gd_depth'] = c['c_gd_depth']; gl['gd_time'] = c['c_gd_ms']; gl['gd_fall'] = c['c_gd_fall']
    gl['gate_noise'] = 0.0
    # VILLAIN clipper (pre-limiter): ceiling with velocity term, clipped share
    gl['clip_ceil'] = c['c_ceil'] * (1 - c['c_clip_vel'] * (1 - max(vel, 0.02)))
    gl['clip_mix'] = c['c_clip_mix'] + (1 - c['c_clip_mix']) * c['c_dry_tanh']
    # E5 wire knobs: not supervised here (weight 0 in prep); E5 defaults (wire ~off)
    gl.update(wire_thr=0.042, wire_level=2e-3, wire_lo=1300.0, wire_hi=7000.0, wire_k=1.3)
    for k, v in gl.items(): t[k] = np.array([v])
    # pretrain weights per mode: sqrt of analytic energy share, floored
    E = t['mode_amp'] ** 2 * t['mode_decay'] / 2
    t['_mode_w'] = np.sqrt(E / E.sum()) + 0.02
    return t

if __name__ == '__main__':
    out = {}
    for s in range(32):
        for vt in VELS:
            rid, vel, t = target(s, vt)
            out[rid] = {'snare': s, 'vel': vel, 'name': man[rid]['name'], 'natural': {k: v.tolist() for k, v in t.items()}}
    json.dump(out, open(os.path.join(HERE, 'targets.json'), 'w'))
    keys = [k for k in out['snare_00_v100']['natural'] if not k.startswith('_')]
    for k in keys:
        a = np.concatenate([np.ravel(o['natural'][k]) for o in out.values()])
        print(f'{k:12s} min {a.min():.4g} p1 {np.percentile(a,1):.4g} med {np.median(a):.4g} p99 {np.percentile(a,99):.4g} max {a.max():.4g}')
