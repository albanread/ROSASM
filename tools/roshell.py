#!/usr/bin/env python3
"""Drive the headless RPCEmu's supervisor prompt and capture output.

The emulator exposes a small JSON-RPC channel: `status`, `type` (inject
keystrokes), `vdu.read` (screen text since an offset), `reset`, `continue`,
`step`, and `snapshot.save`/`snapshot.load`. There is no command method — you
type at the `*` prompt like a person would.

Used as a library by the ObjAsm oracle, or directly:

    roshell.py "Cat" "Show ."
"""
import json
import os
import subprocess
import sys
import time

CWD = r"F:\RISCOSDEV\rpcemu\win32\RPCEmu"
EMU = os.path.join(CWD, "rpcemu-headless.exe")
# Our own snapshot; boot.snap belongs to the user's session.
SNAP = os.path.join(CWD, "rosasm-boot.snap")


class Shell:
    def __init__(self, quiet=True, cwd=None):
        """`cwd` selects which emulator instance to drive, so a farm of copied
        instances can run in parallel. Each needs its own directory: RPCEmu
        writes cmos.ram on exit, and two instances sharing one would race."""
        self.quiet = quiet
        self.cwd = cwd or CWD
        self.p = subprocess.Popen(
            [os.path.join(self.cwd, "rpcemu-headless.exe"), "--rpc"], cwd=self.cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
        )
        self.id = 0

    # ---- raw channel -----------------------------------------------------

    def call(self, method, **params):
        self.id += 1
        req = {"jsonrpc": "2.0", "id": self.id, "method": method}
        if params:
            req["params"] = params
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("emulator closed the channel")
            obj = json.loads(line)
            if obj.get("id") == self.id:
                if "error" in obj:
                    raise RuntimeError(f"{method}: {obj['error']}")
                return obj.get("result")

    def log(self, msg):
        if not self.quiet:
            print(msg, file=sys.stderr)

    # ---- boot ------------------------------------------------------------

    def boot(self, timeout=90):
        """Wait for the `*` supervisor prompt."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            text = self.call("vdu.read").get("text", "")
            if text.rstrip().endswith("*"):
                self.log(f"[cold boot {time.time()-t0:.0f}s]")
                return
            time.sleep(1)
        raise RuntimeError("no supervisor prompt within timeout")

    # ---- commands --------------------------------------------------------

    def cmd(self, command, settle=2.0, max_wait=120.0):
        """Type a command, wait for the prompt to come back, return its output.

        Two quirks of the keyboard injection are worked around here. The first
        character of a burst arrives with the wrong shift state (`Cat` came out
        as `cCat`), so the command is prefixed with a space, which RISC OS
        ignores at a `*` prompt. And `vdu.read`'s `since` parameter did not
        filter reliably, so output is sliced by remembered length instead.
        """
        base = len(self.call("vdu.read").get("text", ""))
        self.call("type", text=" " + command + "\r")

        t0 = time.time()
        last = ""
        stable = 0.0
        while time.time() - t0 < max_wait:
            time.sleep(0.5)
            text = self.call("vdu.read").get("text", "")[base:]
            if text == last:
                stable += 0.5
                # Back at the prompt and nothing changing: done.
                if text.rstrip().endswith("*") and stable >= settle:
                    break
                if stable >= max(settle * 3, 6.0):
                    break
            else:
                stable = 0.0
                last = text
        # Drop the echoed command line and the trailing prompt.
        out = last
        head, sep, rest = out.partition(chr(10))
        if sep and command.split()[0] in head:
            out = rest
        return out.strip().removesuffix("*").rstrip()

    def close(self):
        try:
            self.call("quit")
        except Exception:
            pass
        try:
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def main():
    sh = Shell(quiet=False)
    try:
        sh.boot()
        for c in sys.argv[1:]:
            print(f"*{c}")
            print(sh.cmd(c))
            print()
    finally:
        sh.close()


if __name__ == "__main__":
    main()
