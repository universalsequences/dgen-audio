"""Morph build (A = B) vs the A/B Latent Snare with head/wire taps (make_refs.py): head, wire and out rel RMS.
usage: t_parity.py [BUILD_DIR] [SNARE ...]   (BUILD_DIR relative to $MODAL_AD_ROOT/e8)"""
import sys
from lib8 import *
ref = Instrument("ref/headtap"); new = Instrument(sys.argv[1] if len(sys.argv)>1 else "build/h128tap"); print(new.build_dir)
for nm in sys.argv[2:] or ["SP01"]:
    c = CODES[nm]
    a = render(ref, 1.0, params=zp(c)); b = render(new, 1.0, params={**zp(c), **zp(c, "b")})
    print(nm, "head rel %.3e  wire rel %.3e  out rel %.3e" % (rel(b[:,1], a[:,1]), rel(b[:,2],a[:,2]), rel(b[:,0], a[:,0])))
