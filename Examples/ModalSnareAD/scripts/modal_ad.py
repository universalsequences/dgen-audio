"""Paths shared by the ModalSnareAD / ModalKickAD research scripts.

Code and data live apart. The scripts are checked in under Examples/ModalSnareAD/scripts and
Examples/ModalKickAD/scripts; everything they read and write (renders, targets, runs, staged
instruments) goes under $MODAL_AD_ROOT, mirroring the code layout stage by stage:

    $MODAL_AD_ROOT/data/villain/      VILLAIN Snare dataset      (scripts/dataset/render_villain_dataset.py)
    $MODAL_AD_ROOT/villain_stems/     VILLAIN Snare stem-tap copy (scripts/dataset/make_stems_copy.py)
    $MODAL_AD_ROOT/e1 ... e8/         per-stage outputs           (scripts/e1 ... e8)
    $MODAL_AD_ROOT/kick/              Latent Kick                 (ModalKickAD/scripts)
    $MODAL_AD_ROOT/tools/dgenlisp/    DGenLisp from this checkout (scripts/make_dgenlisp_tool.sh)

Environment:
    MODAL_AD_ROOT        required: the data directory (start from an empty one)
    ESEQ_ROOT            required for anything that renders: an eseq checkout (VILLAIN Snare /
                         VILLAIN Kick sources, tools/audition, the hermetic dgen toolchain)
    ESEQ_DGENLISP_TOOL   the DGenLisp compiler; defaults to $MODAL_AD_ROOT/tools/dgenlisp/DGenLisp
    MODAL_AD_BIN         directory holding the release ModalSnareAD / ModalKickAD binaries
                         (default: this checkout's .build/release)
"""
import os
import sys

SNARE_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
KICK_SCRIPTS = os.path.normpath(os.path.join(SNARE_SCRIPTS, "..", "..", "ModalKickAD", "scripts"))
DGEN_ROOT = os.path.normpath(os.path.join(SNARE_SCRIPTS, "..", "..", ".."))


def _require(name, what):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"{name} is not set: {what}")
    return os.path.abspath(os.path.expanduser(v))


ROOT = _require("MODAL_AD_ROOT", "point it at the data directory these scripts read and write "
                "(see Examples/ModalSnareAD/README.md)")


def stage(name):
    """Data directory of a pipeline stage ("e2", "kick", "kick/morph", ...), created on demand."""
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def code(name):
    """Checked-in code directory of a stage."""
    if name == "kick":
        return KICK_SCRIPTS
    if name.startswith("kick/"):
        return os.path.join(KICK_SCRIPTS, name[len("kick/"):])
    return os.path.join(SNARE_SCRIPTS, name)


def use(*names):
    """Make the scripts of these stages importable (first name wins)."""
    for n in reversed(names):
        p = code(n)
        if p not in sys.path:
            sys.path.insert(0, p)


def eseq_root():
    return _require("ESEQ_ROOT", "point it at an eseq checkout (VILLAIN instruments, tools/audition)")


def dgenlisp():
    tool = os.environ.get("ESEQ_DGENLISP_TOOL") or os.path.join(ROOT, "tools", "dgenlisp", "DGenLisp")
    if not os.path.exists(tool):
        sys.exit(f"no DGenLisp at {tool}: run Examples/ModalSnareAD/scripts/make_dgenlisp_tool.sh "
                 "or set ESEQ_DGENLISP_TOOL")
    return os.path.abspath(tool)


def audition(pinned=False):
    """Make eseq's tools/audition importable (`from audition import Instrument`) with eseq's
    hermetic clang/lld stage. The Latent instruments need this checkout's DGenLisp (the default);
    pinned=True keeps ESEQ_DGENLISP_TOOL if set, else eseq's own fetched compiler, which is what
    rendered the VILLAIN datasets (scripts/fetch_dgenlisp.sh in eseq installs it)."""
    repo = eseq_root()
    if pinned:
        os.environ.setdefault("ESEQ_DGENLISP_TOOL", os.path.join(repo, "crates/sequencer/tools/DGenLisp-macos-arm64"))
    else:
        os.environ["ESEQ_DGENLISP_TOOL"] = dgenlisp()
    os.environ.setdefault("ESEQ_DGEN_TOOLCHAIN_ROOT", os.path.join(repo, "crates/sequencer/tools/dgen-toolchain"))
    p = os.path.join(repo, "tools", "audition")
    if p not in sys.path:
        sys.path.insert(0, p)
    return repo


def binary(name):
    """Release build of a dgen example (ModalSnareAD / ModalKickAD)."""
    d = os.environ.get("MODAL_AD_BIN") or os.path.join(DGEN_ROOT, ".build", "release")
    p = os.path.join(d, name)
    if not os.path.exists(p):
        sys.exit(f"{p} not found: swift build -c release --product {name} (or set MODAL_AD_BIN)")
    return p
