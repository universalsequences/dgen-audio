#!/usr/bin/env python3
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent


def request(sock_path: Path, command: str) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(sock_path))
        client.sendall((command + "\n").encode())
        data = b""
        while b"\n" not in data:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk
    return data.decode().strip()


def manifest(default: float) -> dict:
    return {
        "version": 3,
        "processAbi": "dgen-host-abi-v1",
        "dylib": "fixture.so",
        "sampleRate": 44100,
        "maxFrameCount": 8,
        "voiceCount": 1,
        "totalMemorySlots": 16,
        "params": [{"name": "x", "cellId": 0, "cellSpan": 2, "default": default}],
        "inputs": [],
        "outputs": [{"channel": 0, "name": "audio"}],
        "tensorInitData": [{"offset": 10, "data": [3.0, 4.0]}],
    }


def main() -> None:
    invalid_block = subprocess.run(
        [str(ROOT / "dgen-live"), "--no-audio", "--block-size", "2"],
        capture_output=True, text=True,
    )
    assert invalid_block.returncode == 2
    assert "multiple of 4" in invalid_block.stderr

    with tempfile.TemporaryDirectory(prefix="dgen-live-test-") as temp:
        temp = Path(temp)
        sock = temp / "live.sock"
        os.symlink(ROOT / "tests" / "fixture.so", temp / "fixture.so")
        first = temp / "first.json"
        second = temp / "second.json"
        bad = temp / "bad.json"
        first.write_text(json.dumps(manifest(2.5)))
        second.write_text(json.dumps(manifest(1.0)))
        invalid = manifest(1.0)
        invalid["tensorInitData"] = [{"offset": 1023, "data": [1.0, 2.0]}]
        bad.write_text(json.dumps(invalid))

        process = subprocess.Popen(
            [str(ROOT / "dgen-live"), "--no-audio", "--socket", str(sock),
             "--sample-rate", "44100", "--block-size", "4"],
            stderr=subprocess.PIPE, text=True
        )
        try:
            deadline = time.time() + 5
            while not sock.exists() and time.time() < deadline:
                if process.poll() is not None:
                    raise AssertionError(process.stderr.read())
                time.sleep(0.02)
            assert request(sock, "PING") == "OK pong"
            assert json.loads(request(sock, "STATUS")[3:]) == {
                "xruns": 0, "partialWrites": 0
            }
            assert request(sock, f"LOAD {first}").startswith("OK loaded ")
            assert request(sock, "RENDER 2").startswith(
                "ERR frame count must be divisible by 4"
            )
            values = json.loads(request(sock, "RENDER 4")[3:])
            assert values == [9.0] * 4, values  # 2.5 + 2.5 + 3 + context(1)
            assert request(sock, f"LOAD {second}").startswith("OK loaded ")
            values = json.loads(request(sock, "RENDER 4")[3:])
            assert values == [6.0] * 4, values
            assert request(sock, f"LOAD {bad}").startswith("ERR tensorInitData")
            assert request(sock, "STOP") == "OK stopped"
            assert request(sock, "RENDER 4") == "ERR no patch loaded"
            assert request(sock, "QUIT") == "OK quitting"
            assert process.wait(timeout=5) == 0
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
