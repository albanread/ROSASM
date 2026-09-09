# rosasm

An ObjAsm-compatible assembler for the RISC OS 5 sources, written in Rust.

ObjAsm is Acorn's macro assembler. It is also 52% of RISC OS 5 by line count —
630,460 lines of assembler against 579,893 lines of C — and it runs only on
RISC OS. So a RISC OS build on a development machine needs an assembler that
speaks ObjAsm's language, and there isn't one. This is that assembler.

The division of labour is deliberate:

```
ObjAsm source → expand → lower → legalize → LLVM → AOF
```

Everything above the mnemonic is ours — the macro language, conditional
assembly, the expression evaluator, the symbol table, layout. LLVM's
integrated assembler is used only to turn a legal UAL instruction into four
bytes. Everything below the object file is ours again, because the RISC OS
linker reads AOF and LLVM has never heard of it.

Target: Raspberry Pi 4, Cortex-A72, ARMv8-A in AArch32, with NEON in place of
the FPA the sources were written for.

## What is here

| | |
|---|---|
| `source`, `lex`, `vocab` | reading the sources as ObjAsm does: Latin-1, column-sensitive fields, 47 directives |
| `symtab`, `expr` | three value types, unsigned arithmetic, 21 operators |
| `expand` | macros, conditional assembly, two passes, `GET` and `LNK` |
| `layout` | areas, location counters, local-label scopes |
| `lower`, `legalize` | pre-UAL to UAL, and the forms that have no UAL spelling at all |
| `reloc`, `aof`, `elfread` | the object file, and the fixups the linker needs |
| `listing` | ObjAsm's listing format, for comparing against the real thing |

Legalization is the stage that decides what an Acorn instruction can *become*,
because some forms have no UAL spelling at all. `ADR` and `ADRL` are
pseudo-instructions whose targets are ours to resolve, so they are expanded
here; `LDR Rd,=value` becomes a `MOV` or an `MVN` where one will do;
`TEQP`, `TSTP`, `CMPP` and `CMNP` wrote the PSR on a 26-bit ARM and have no
32-bit equivalent at all.

## Building

Needs a Rust toolchain and clang, used as the encoder:

```bash
cargo build --release
```

The encoder's path is `CLANG` in `src/bin/rosasm.rs`.

## Using it

```bash
rosasm <source> -o <object> [-I dir]... [-PD "Machine SETS \"RPi\""]...
```

`aofdump` shows what came out, in roughly the shape the DDE's `decaof` prints
it:

```bash
aofdump <object> -x
```

```
** Area 0: Demo$$Code
   size &1C (28), alignment 2^2, attributes: CODE, READONLY, APCS-32
   &0000: E59F0010  ....
   &0004: E3500000  ..P.
   &0008: 0A000000  ....
   &000C: EBFFFFFB  ....
   reloc &000C: instruction, by symbol 1, pc-relative, 1 instruction(s)

** Symbol table
   Go                       Demo$$Code + &0          defined, global
   Elsewhere                &0                       reference, global
```

## Checking it against the real ObjAsm

The corpus is the oracle. `tools/` holds the harness:

- `roshell.py` drives a headless RPCEmu at the RISC OS `*` prompt.
- `corpus_diff.py` stages each translation unit, assembles it with the real
  ObjAsm to a listing and with `roslist` here, and compares the two. It
  replaces `amu` as the build driver.
- `farm.py` runs N emulator instances in parallel.
- `codegen_sweep.py` needs no emulator: it assembles the whole corpus here and
  reports, grouped by cause, what stopped the rest.

## Where it has got to

Against the RISC OS 5 sources for the BCM2835 target:

| | |
|---|---:|
| translation units | 252 |
| reaching an AOF object | 173 |
| code emitted | 55,496 bytes in 212 areas |

Relocations are emitted for branches, literal loads and data words, by area
base or by symbol, with same-area references resolved here rather than left to
the linker.

What is left is mostly the FPA instruction set — `LDFD`, `STFE`, `CMF`, `MUFE`
and the rest — which this target replaces with VFP and NEON, plus a tail of
sources that need headers only the build's `export_hdrs` step produces.

## Licence

Not yet declared. The surrounding tree licenses its own work under Apache-2.0
(see `LICENSING.md` there), which is the intended choice here too — but this
repository carries no `LICENSE` file yet.
