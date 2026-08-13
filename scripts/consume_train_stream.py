#!/usr/bin/env python3
"""Mock eseq host for `dgenlisp train` (patch-learn-spec rev 1, §3-§5).

Spawns the CLI, consumes its NDJSON stdout stream LIVE (line by line), and
strictly validates the protocol:

  - every stdout line is valid JSON with a known "type" and the exact
    required keys for that type (unknown type or malformed JSON = failure)
  - the first event is "plan"
  - exactly one terminal event ("result" or "error"), and it is the last line
  - exit code is 0 iff the terminal event was "result"
  - artifact paths referenced by checkpoint/result events exist on disk and
    live inside the --job-dir
  - result.json exists and matches the streamed result event

Usage:
  consume_train_stream.py <dgenlisp-binary> <train args...>

Exits 0 if the stream is protocol-clean, 1 with a diagnostic otherwise.
"""

import json
import os
import subprocess
import sys
from typing import NoReturn

REQUIRED_KEYS = {
    "plan": {"type", "learnable", "frozen", "unsupported", "seed_echo",
             "pitch_hz", "gate_frames", "crop_frames"},
    "stage": {"type", "name", "total"},
    "epoch": {"type", "epoch", "total", "loss", "params"},
    "checkpoint": {"type", "epoch", "wav"},
    "result": {"type", "improvement_pct", "abs_distance", "basin_check",
               "deltas", "final_wav"},
    "error": {"type", "message"},
}

TERMINAL = {"result", "error"}


def fail(msg) -> NoReturn:
    print(f"PROTOCOL VIOLATION: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        fail("usage: consume_train_stream.py <dgenlisp-binary> <train args...>")

    binary, args = sys.argv[1], sys.argv[2:]
    if args[0] != "train":
        args = ["train"] + args
    job_dir = None
    for i, a in enumerate(args):
        if a == "--job-dir" and i + 1 < len(args):
            job_dir = args[i + 1]
    if job_dir is None:
        fail("--job-dir not present in args (mock host needs it to check artifacts)")
    job_dir = os.path.abspath(job_dir)

    proc = subprocess.Popen(
        [binary] + args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    assert proc.stdout is not None and proc.stderr is not None
    events = []
    for line in proc.stdout:  # live, line-by-line
        line = line.rstrip("\n")
        if not line:
            fail("blank line on stdout")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            fail(f"malformed JSON on stdout: {line!r} ({e})")
        if not isinstance(obj, dict) or "type" not in obj:
            fail(f"event without type: {line!r}")
        etype = obj["type"]
        if etype not in REQUIRED_KEYS:
            fail(f"unknown event type {etype!r}: {line!r}")
        missing = REQUIRED_KEYS[etype] - set(obj.keys())
        if missing:
            fail(f"{etype} event missing keys {sorted(missing)}: {line!r}")
        extra = set(obj.keys()) - REQUIRED_KEYS[etype]
        if extra:
            fail(f"{etype} event has unexpected keys {sorted(extra)}: {line!r}")
        events.append(obj)

    proc.wait()
    stderr_tail = proc.stderr.read()[-2000:]

    if not events:
        fail(f"no events on stdout (exit {proc.returncode}); stderr tail:\n{stderr_tail}")
    if events[0]["type"] != "plan":
        fail(f"first event must be plan, got {events[0]['type']}")

    terminals = [e for e in events if e["type"] in TERMINAL]
    if len(terminals) != 1:
        fail(f"expected exactly one terminal event, got {len(terminals)}")
    if events[-1]["type"] not in TERMINAL:
        fail(f"last event must be terminal, got {events[-1]['type']}")

    terminal = events[-1]
    if terminal["type"] == "result" and proc.returncode != 0:
        fail(f"result emitted but exit code is {proc.returncode}")
    if terminal["type"] == "error" and proc.returncode == 0:
        fail("error emitted but exit code is 0")

    # Artifact checks.
    def check_artifact(path, label):
        if not os.path.isfile(path):
            fail(f"{label} references missing file: {path}")
        if not os.path.abspath(path).startswith(job_dir + os.sep):
            fail(f"{label} artifact escaped the job dir: {path}")

    for e in events:
        if e["type"] == "checkpoint":
            check_artifact(e["wav"], "checkpoint")
    if terminal["type"] == "result":
        check_artifact(terminal["final_wav"], "result")
        for name in ("lowered.lisp", "final.wav", "result.json"):
            if not os.path.isfile(os.path.join(job_dir, name)):
                fail(f"job dir missing required artifact {name}")
        with open(os.path.join(job_dir, "result.json")) as f:
            stored = json.load(f)
        if stored != terminal:
            fail("result.json does not match the streamed result event")
        # deltas shape: {name: {from, to}}
        for name, d in terminal["deltas"].items():
            if set(d.keys()) != {"from", "to"}:
                fail(f"delta for {name} has wrong shape: {d}")

    print(f"OK: {len(events)} events, terminal={terminal['type']}, exit={proc.returncode}")
    sys.exit(0)


if __name__ == "__main__":
    main()
