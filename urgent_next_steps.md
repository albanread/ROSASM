# rosasm — urgent next steps (2026-09-10)

From the review of 2026-09-10. The head build was made and its 319 tests run,
the corpus sweep was reproduced on it, and every encoder rejection in that
sweep was traced back to the source line that caused it (the driver cannot do
that yet, see §1). Numbers below are from that run, not from the committed
reports.

What rosasm is: the ObjAsm language, layout and object format in Rust, with
clang used to encode single instructions. It works: 212 of the 241 units in
the BCM2835 build reach an AOF object. What it is not yet: an assembler a ROM
can be built with, and the gap is enumerable. This list is that enumeration,
in the order that makes the corpus numbers trustworthy first.

| Measured on the head build | |
|---|---:|
| units in the BCM2835 build | 241 |
| assembling to an object | 212 (88.0%) |
| code bytes emitted | 240,596 in 254 areas |
| instructions emitted as zero words | 220, all FPA extended precision and transcendentals |
| failures | 29: 13 outside the build, 10 encoder rejections, 3 assertions, 3 missing generated sources |
| byte-identical to ObjAsm | 4, in an oracle run that compared 18 units before its ObjAsm side died |
| sweep time, 8 jobs | 2 minutes |

## 0. An object with wrong words in it must not exit 0

`src/bin/rosasm.rs` prints a line and emits zero words for an unsupported
instruction, an `LDR =` whose pool is out of reach, an unhandled relocation
type, or a branch out of range, and then writes the object and exits 0. A zero
word is `ANDEQ r0, r0, r0`: it executes and does nothing, so a ROM built this
way runs past the missing instruction silently. The sweep only knows because
it scrapes stderr.

- Default: any of those is a failure, exit 1, no object.
- Opt-in `--allow-unencodable`: emit `UDF #n` instead of a zero word, with n
  indexing a table printed at the end, so a missing instruction traps loudly
  at run time and names itself.
- The sweep's "instructions with no equivalent" report keeps working under
  the opt-in; without it those units count as failures, which is the truth.

## 1. Encoder errors must point at the source

clang's diagnostics name a temp file and a line in it. Without `--keep-temps`
and a hand-made join there is no way back to the source, which the design
document called out as the daily pain if bolted on late. The driver already
has everything needed: every instruction is labelled `__ros<i>` in the
generated text, and `--map` lists index, address, area, file and line. On an
encoder failure, read its output, walk back from each error line to the
nearest `__ros` label, and print `file:line: <original text>` with clang's
message underneath. Thirty lines; it turned the ten rejections below into
this list in one run.

## 2. The ten encoder rejections, by cause

Traced with the join above; each is a small, specific fix.

- **`#imm, rotation` misfires on a register** (5 sites, `BCMVideo` via
  `GVOverlay.s:313, 314, 981, 983, 986`). In `encoder_operands`, the last
  operand after `#imm,` is taken as a rotation when it evaluates to an even
  number under 32. Register names evaluate to their numbers, so
  `SSAT r2, #16, r2` becomes `#0x10, 2` and `SSAT lr, #16, lr` becomes
  `#0x10, 14`. An odd register passes untouched, so this is one operand away
  from silent. Apply the rotation form only to data-processing mnemonics, and
  never when the token is a register name or alias.
- **`UND #0` is not lowered** (`HWPointer.s:457`, `TVService.s:278`). UAL
  spells it `UDF #imm`.
- **`TEQNEP` reaches the encoder** (`DA_HostFS:214`). `is_psr_form` looks
  for `P` immediately after the stem; a condition in between hides it. Match
  `<cmp><cond>P` too, so it is refused as the 26-bit form it is.
- **FPA float literals** (`cl_body.s:1330`, reached from six RISC_OSLib
  units: `cl_obj_r`, `cl_obj_m`, `cl_mod_r`, `k_obj_r`, `k_obj_m`,
  `k_mod_r`). `LDFNES f0, =-0.0` becomes `VLDRNE s0, =-0.0`, which clang
  rejects. A float literal needs the bit pattern of the value in the literal
  pool and a `VLDR` from there, in the pool machinery the expander already
  has for integers. These six are the C runtime's assembler side, so the C
  track needs them.
- **Cross-area `LDR`/`STR` of a label plus a field** (`NetFiler:1467, 1468,
  1537, 1540, 1542, 1545`). `STR r14, mm_display + mi_submenu` names a label
  in another area; `fold_address` leaves it, and clang cannot relocate it.
  Encode it as the ADR-at-import case is encoded now: offset zero in the
  instruction and an AOF relocation by the target area, with the field's
  value folded into the addend.
- **A string immediate** (`PipeFS:1996`): `CMPNE R14, #"\\"`. One site;
  decide what ObjAsm makes of a two-character string as an immediate before
  implementing it.

## 3. One assertion is a layout question, not an environment one

`HAL_BCM2835/s/SPI:79` asserts `. - 64 = HALDeviceSize` and gets 0 against
64: the location counter is 64 short of where ObjAsm has it at that line. HAL
device blocks are laid out to the byte and the HAL is the first thing a ROM
runs, so find the directive that moved `.` differently before trusting any
HAL unit. `SCSIDriver:365` (`FormRevisionSrc = 4`, got 1) reads like a build
option the sweep does not pass; `Fonts` times out rather than asserting.

## 4. Finish an oracle run, then publish the number

The design's own metric is units byte-identical to ObjAsm, of 241. It has not
been measured: the last run's log shows the ObjAsm side failing every unit
from about unit 30 onwards (130 of 150 by the end), which is the emulator
side breaking, not ObjAsm rejecting sources. `tools/aofdiff.py --all` needs a
health check that stops after a few consecutive oracle-side failures and says
what it saw, and a way to resume. Until a run completes, the honest figure is
4 identical of 18 compared.

## 5. Driver and harness hygiene

- The clang path is a constant in the driver; the sweep can override
  `ROSASM` but nothing can override `CLANG`. Take an environment variable
  and a `--clang` option.
- Temp files are named by pid only; two runs in one process tree can
  collide. Add a random suffix.
- `tools/aofdiff.py` carries an uncommitted change (per-row report writing,
  19 lines). It parses and looks right; commit it.
- `src/lower.rs` has a duplicated doc-comment line above `lower_instruction`.

## 6. From an object file to a ROM

Nothing produced by rosasm has yet been linked. The next proof after the
fixes above is one component, linked by the DDE's own `link` under the
emulator with rosasm's objects substituted for ObjAsm's, in a ROM that boots.
Byte-identical objects that fail to boot would mean the comparison is
normalising away something real, which is why the design document put "link
and boot" as the second-order check. Do it for one small module first.

## 7. Verified on 2026-09-10, so nobody re-verifies it

| Check | Result |
|---|---|
| `cargo build --release` | clean, zero warnings |
| `cargo test` | 292 unit tests and 27 integration tests pass |
| `tools/codegen_sweep.py` on the head build | 241 units, 212 objects, 220 zero words, 29 failures |
| `--cpu`, target, encoder | clang 22 as `--target=arm-none-eabi -mcpu=cortex-a72 -mfpu=neon-fp-armv8` |
| the nine ADRL zero words from the committed report | gone since `cbec42f` |

Order: 0 and 1 first, because they decide whether the sweep can be believed;
then §2 in the order given; then the SPI assertion; then a completed oracle
run; then one component linked and booted.
