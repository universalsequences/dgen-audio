#!/usr/bin/env python3
"""Target / previous shipping voice / new shipping voice, including real tails.

Both synths are rendered through the compiled eseq DSP. Writes raw and explicitly
RMS-matched listening reels, plus the exact gains and independent comparison.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import signal
from scipy.io.wavfile import write

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import compare


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eseq-root',type=Path,required=True)
    ap.add_argument('--old-instrument',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--target',type=Path,default=Path('Assets/r8-kick03.wav'))
    args=ap.parse_args()
    sys.path.insert(0,str(args.eseq_root/'tools/audition'))
    from audition import Instrument
    sr=48000
    target,sr0=compare.read_wav(args.target)
    target=signal.resample(target,round(len(target)*sr/sr0)).astype(np.float32)
    active_frames=len(target)
    target=np.pad(target,(0,round(.7*sr)-len(target)))
    sounds=[target]
    for path in (args.old_instrument,args.eseq_root/'content/instruments/Drums/R8 Kick 03'):
        inst=Instrument(str(path),sample_rate=sr)
        subprocess.run([sys.executable,str(args.eseq_root/'tools/audition/check_fusion.py'),str(Path(inst.build_dir)/'patch.c')],check=True)
        y,_=inst.render(.7,pitch=261.63,vel=1.)
        assert np.isfinite(y).all()
        sounds.append(y)
    names=('target','old','new')
    gap=np.zeros(round(.2*sr),dtype=np.float32)
    raw=np.concatenate([part for _ in range(3) for y in sounds for part in (y,gap)])
    raw_gain=min(1.,.95/float(np.max(np.abs(raw))))
    write(str(args.run/'ab-target-old-new.wav'),sr,(raw*raw_gain).astype(np.float32))
    rms=[float(np.sqrt(np.mean(y[:active_frames]**2))) for y in sounds]
    matched=[y*rms[0]/v for y,v in zip(sounds,rms)]
    matched_gain=min(1.,.95/max(float(np.max(np.abs(y))) for y in matched))
    reel=np.concatenate([part for _ in range(3) for y in matched for part in (y,gap)])
    write(str(args.run/'ab-target-old-new-level-matched.wav'),sr,(reel*matched_gain).astype(np.float32))
    n=round(.24*sr)
    hp=lambda y:compare.capture_highpass(y[:n],sr,30.)
    report={'order':names,'sampleRate':sr,'activeFrames':active_frames,
            'rawReelCommonGain':raw_gain,'levelMatchedGains':{name:matched_gain*rms[0]/v for name,v in zip(names,rms)},
            'rawRms':dict(zip(names,rms)),
            'gate240ms':{name:compare.mrstft(hp(target),hp(y)) for name,y in zip(names,sounds)}}
    (args.run/'ab-report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,y in zip(names,sounds):
        write(str(args.run/f'ab-{name}-full.wav'),sr,y.astype(np.float32))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
