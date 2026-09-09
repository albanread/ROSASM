#!/usr/bin/env python3
"""Assemble the whole corpus with rosasm and tally what code generation does.

    codegen_sweep.py <RiscOS dir> [--limit N] [--jobs N] [--out report.txt]

No emulator: this is our own pipeline end to end -- expand, lower, legalize,
encode, write AOF -- run over every unit `corpus_diff.py` recognises. The
differential harness answers "does our listing match ObjAsm's"; this answers
"how much of the corpus reaches an object file at all, and what stops the
rest".

Failures are grouped by cause rather than counted, because the interesting
output is the shortlist of things to fix next.
"""
import argparse
import collections
import concurrent.futures
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_diff  # noqa: E402
from corpus_diff import BUILD_VARS, units  # noqa: E402

ROSASM = r"F:\RISCOSDEV\rosasm\target\release\rosasm.exe"
PD = [f'{k} SETS "{v}"' for k, v in BUILD_VARS]


def classify(err):
    """Group a failure by cause, keeping the part that identifies it."""
    m = re.search(r"cannot find ('?[^'\n]+'?)", err)
    if m:
        return "cannot find " + m.group(1).strip("'")
    for pat, label in [
        (r"unclosed conditional", "unclosed conditional"),
        (r"the encoder rejected", "encoder rejected the lowered assembly"),
        (r"assertion failed", "assertion failed"),
        (r"unknown AREA attribute", "unknown AREA attribute"),
        (r"undefined symbol ([^\s,]+)", "undefined symbol"),
    ]:
        if re.search(pat, err):
            return label
    first = next((l for l in err.splitlines() if l.strip()), "")
    return first.strip()[:80] or "failed with no message"


# Warnings the driver prints per instruction. These do not stop the object
# being written -- the instruction becomes a zero word and the addresses
# around it stay put -- but they say what would not execute.
UNSUPPORTED = re.compile(r"rosasm: [^\n]*?:\d+: ([A-Z][A-Z0-9.]*)[: ]")
UNHANDLED_RELOC = re.compile(r"unhandled relocation type (\d+)")


def assemble(unit):
    """Run the whole pipeline on one unit. Returns a result dict."""
    comp = os.path.dirname(os.path.dirname(unit))
    args = [ROSASM, unit, "-I", comp, "-I", os.path.join(comp, "hdr"), "-I", HDRROOT[0]]
    for pd in PD:
        args += ["-PD", pd]
    with tempfile.TemporaryDirectory() as tmp:
        args += ["-o", os.path.join(tmp, "out.o")]
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return {"unit": unit, "ok": False, "why": "timed out"}
        except OSError as e:
            return {"unit": unit, "ok": False, "why": f"cannot run rosasm: {e}"}
        out, err = p.stdout, p.stderr
        ok = p.returncode == 0
        size = os.path.getsize(os.path.join(tmp, "out.o")) if ok else 0
    m = re.search(r"(\d+) bytes in (\d+) area", out)
    return {
        "unit": unit,
        "ok": ok,
        "why": None if ok else classify(err),
        "bytes": int(m.group(1)) if m else 0,
        "areas": int(m.group(2)) if m else 0,
        "object": size,
        "unsupported": [m.group(1) for m in UNSUPPORTED.finditer(err)],
        "relocs": [m.group(1) for m in UNHANDLED_RELOC.finditer(err)],
    }


HDRROOT = [None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out")
    a = ap.parse_args()

    # The emulator's disc is not involved here, so the export root goes
    # somewhere local rather than into hostfs.
    corpus_diff.HDRROOT = os.path.join(tempfile.gettempdir(), "rosasm-hdrs")
    HDRROOT[0] = corpus_diff.build_export_root(a.root)
    us = units(a.root)
    if a.limit:
        us = us[: a.limit]
    print(f"{len(us)} units, {a.jobs} jobs", flush=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        futures = {pool.submit(assemble, u): u for u in us}
        for n, f in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(f.result())
            if n % 25 == 0:
                print(f"  {n}/{len(us)}", flush=True)

    ok = [r for r in results if r["ok"]]
    lines = []
    lines.append(f"units            {len(results)}")
    lines.append(f"assembled        {len(ok)} ({100 * len(ok) / max(1, len(results)):.1f}%)")
    lines.append(f"code bytes       {sum(r['bytes'] for r in ok)}")
    lines.append(f"areas            {sum(r['areas'] for r in ok)}")

    unsupported = collections.Counter(m for r in results for m in r.get("unsupported", []))
    if unsupported:
        lines.append("")
        lines.append(
            f"instructions with no equivalent on this target "
            f"({sum(unsupported.values())} in all, each a zero word):"
        )
        for m, n in unsupported.most_common(20):
            lines.append(f"  {n:5d}  {m}")

    relocs = collections.Counter(m for r in results for m in r.get("relocs", []))
    if relocs:
        lines.append("")
        lines.append("relocation types not translated:")
        for m, n in relocs.most_common(20):
            lines.append(f"  {n:5d}  ELF type {m}")

    failed = collections.Counter(r["why"] for r in results if not r["ok"])
    if failed:
        lines.append("")
        lines.append("failures by cause:")
        for why, n in failed.most_common(40):
            lines.append(f"  {n:5d}  {why}")

    report = "\n".join(lines)
    print()
    print(report)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(report + "\n\n")
            for r in sorted(results, key=lambda r: r["unit"]):
                mark = "ok  " if r["ok"] else "FAIL"
                f.write(f"{mark} {r['unit']}  {r['why'] or str(r['bytes']) + ' bytes'}\n")
        print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
