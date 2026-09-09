//! The FPA instruction set, and what of it VFP can express.
//!
//! The sources were written for the Floating Point Accelerator: eight
//! registers `f0`-`f7`, and instructions that name the precision they work in.
//! A Cortex-A72 has no FPA. It has VFP and Advanced SIMD, so the FPA
//! instructions have to become VFP ones — where that is possible at all.
//!
//! It is not always possible, and the difference matters:
//!
//! * **Single and double** map exactly. `ADFS` rounds to single and so does
//!   `VADD.F32`; `LDFD` moves eight bytes and so does `VLDR` on a `d`
//!   register. These are translations, and 270 of the corpus's 598 FPA sites
//!   are of this kind.
//! * **Extended precision** has no VFP format at all. The FPA's `E` operations
//!   work to about 64 bits of mantissa and store 12 bytes; VFP's widest is a
//!   64-bit double with 53. Rewriting `STFE` as `VSTR` would silently drop
//!   bits, and the code that uses it — `fma`, `atan2`, `log1p` in the C
//!   library's `mathasm` — exists precisely to not drop them. Those routines
//!   need porting, not translating, so they are refused here by name.
//! * **Transcendentals** (`SIN`, `LOG`, `POW`, ...) were FPA instructions and
//!   are library calls everywhere else. There is nothing to translate to.
//! * **`LFM`/`SFM`** transfer registers in the FPA's internal 12-byte format,
//!   so they are the extended-precision problem again.
//! * **Packed decimal** (`LDFP`, `STFP`) has no equivalent in anything.
//!
//! ## Which register is `f3`?
//!
//! An FPA register holds one value, so the mapping has to give each `f<n>` one
//! place to live: `f<n>` becomes `d<n>`, and a single-precision operation uses
//! `s<2n>`, the low half of that same `d<n>`. Nothing collides — `s1`, `s3`,
//! `s5` and so on are left unused — and a register written as a single and
//! read as a single comes back unchanged, which is what the convertible subset
//! does. Code that writes one precision and reads another is doing a deliberate
//! conversion, and in this corpus that always involves `E`, which is refused.
//!
//! ## Sizes change
//!
//! Every FPA instruction is one word. `CMF` is not: the FPA set the ARM flags
//! directly, while `VCMP` sets the FPSCR and a following `VMRS` copies them
//! across. So a compare becomes two instructions, and the location counter has
//! to know that before the encoder ever runs — which is why `words` exists and
//! why layout consults it during expansion.

use crate::legalize::Legalized;

/// The format an FPA instruction works in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision {
    Single,
    Double,
    /// About 64 bits of mantissa, 12 bytes in memory. No VFP equivalent.
    Extended,
    /// Packed decimal. No equivalent anywhere.
    Packed,
}

impl Precision {
    fn from_letter(c: char) -> Option<Precision> {
        match c {
            'S' => Some(Precision::Single),
            'D' => Some(Precision::Double),
            'E' => Some(Precision::Extended),
            'P' => Some(Precision::Packed),
            _ => None,
        }
    }

    /// The VFP type suffix, for the precisions VFP has.
    fn suffix(self) -> Option<&'static str> {
        match self {
            Precision::Single => Some("F32"),
            Precision::Double => Some("F64"),
            _ => None,
        }
    }

    /// Bytes one register occupies in memory.
    fn width(self) -> u32 {
        match self {
            Precision::Single => 4,
            Precision::Double => 8,
            Precision::Extended => 12,
            Precision::Packed => 12,
        }
    }
}

/// A decoded FPA mnemonic.
#[derive(Debug, Clone, PartialEq)]
pub struct Fpa {
    pub stem: &'static str,
    /// The ARM condition, uppercase and canonical, or empty.
    pub cond: String,
    pub precision: Option<Precision>,
    /// `P`, `M` or `Z` — round towards plus infinity, minus infinity or zero.
    /// VFP has no per-instruction rounding mode, so any of these is refused.
    pub rounding: Option<char>,
    /// `CMFE`/`CNFE`: raise an exception on unordered operands.
    pub exception: bool,
}

/// Every FPA stem, longest first so a prefix never wins over a longer match.
const STEMS: [&str; 41] = [
    "LDF", "STF", "LFM", "SFM", "ADF", "SUF", "RSF", "MUF", "DVF", "RDF", "RMF", "FML", "FDV",
    "FRD", "POW", "RPW", "POL", "MVF", "MNF", "ABS", "RND", "SQT", "LOG", "LGN", "EXP", "SIN",
    "COS", "TAN", "ASN", "ACS", "ATN", "URD", "NRM", "CMF", "CNF", "FLT", "FIX", "RFS", "WFS",
    "RFC", "WFC",
];

/// The register-transfer instructions, which take an ARM register and no
/// precision at all.
const STATUS: [&str; 4] = ["RFS", "WFS", "RFC", "WFC"];

/// Stems with no VFP instruction to become: these were FPA operations and are
/// library calls on every other architecture.
const TRANSCENDENTAL: [&str; 12] = [
    "POW", "RPW", "POL", "LOG", "LGN", "EXP", "SIN", "COS", "TAN", "ASN", "ACS", "ATN",
];

/// Addressing modes an `LFM`/`SFM` may carry, as `LDM`/`STM` do.
const BLOCK_MODES: [&str; 6] = ["IA", "DB", "FD", "EA", "FA", "ED"];

const CONDS: [&str; 17] = [
    "EQ", "NE", "CS", "CC", "MI", "PL", "VS", "VC", "HI", "LS", "GE", "LT", "GT", "LE", "AL",
    "HS", "LO",
];

fn canonical_cond(c: &str) -> &'static str {
    match c {
        "HS" => "CS",
        "LO" => "CC",
        "EQ" => "EQ",
        "NE" => "NE",
        "CS" => "CS",
        "CC" => "CC",
        "MI" => "MI",
        "PL" => "PL",
        "VS" => "VS",
        "VC" => "VC",
        "HI" => "HI",
        "LS" => "LS",
        "GE" => "GE",
        "LT" => "LT",
        "GT" => "GT",
        "LE" => "LE",
        _ => "AL",
    }
}

/// Split a mnemonic into stem, condition, precision and rounding.
///
/// The sources write the condition before the precision — `MVFNEE` is `MVF`
/// conditional on `NE` in extended precision — but a rounding mode follows the
/// precision, so `ADFSP` is `ADF` in single precision rounding to plus
/// infinity. Reading the precision first and falling back to a condition
/// resolves the one ambiguous shape: `MVFPLS` cannot be packed-then-`LS`,
/// because `LS` is not a rounding mode, so it is `PL` then single.
pub fn parse(mnemonic: &str) -> Option<Fpa> {
    let up = mnemonic.to_ascii_uppercase();
    let stem = STEMS.iter().find(|s| up.starts_with(**s))?;
    let rest = &up[stem.len()..];

    // The comparisons take no precision: they compare whatever the registers
    // hold. Their only suffix is `E`, and the condition may sit either side of
    // it -- ARM's own documentation writes it both ways.
    if matches!(*stem, "CMF" | "CNF") {
        let (mut rest, mut exception) = (rest, false);
        if let Some(r) = rest.strip_prefix('E') {
            rest = r;
            exception = true;
        }
        let mut cond = String::new();
        if rest.len() >= 2 && CONDS.contains(&&rest[..2]) {
            cond = canonical_cond(&rest[..2]).to_string();
            rest = &rest[2..];
        }
        if let Some(r) = rest.strip_prefix('E') {
            rest = r;
            exception = true;
        }
        if !rest.is_empty() {
            return None;
        }
        return Some(Fpa { stem, cond, precision: None, rounding: None, exception });
    }

    // Reading and writing the FPA's status and control words takes an ARM
    // register and a condition, and nothing else.
    if STATUS.contains(stem) {
        let mut cond = String::new();
        let mut rest = rest;
        if rest.len() >= 2 && CONDS.contains(&&rest[..2]) {
            cond = canonical_cond(&rest[..2]).to_string();
            rest = &rest[2..];
        }
        if !rest.is_empty() {
            return None;
        }
        return Some(Fpa { stem, cond, precision: None, rounding: None, exception: false });
    }

    // The multiple-register transfers carry an addressing mode where the
    // others carry a precision -- `SFMNEFD` is `SFM`, conditional on `NE`,
    // full-descending -- so they are decoded separately. Everything about
    // them is refused later; this only has to recognise them.
    if matches!(*stem, "LFM" | "SFM") {
        let mut rest = rest;
        let mut cond = String::new();
        if rest.len() >= 2 && CONDS.contains(&&rest[..2]) {
            cond = canonical_cond(&rest[..2]).to_string();
            rest = &rest[2..];
        }
        if rest.len() == 2 && BLOCK_MODES.contains(&rest) {
            rest = "";
        }
        if !rest.is_empty() {
            return None;
        }
        return Some(Fpa { stem, cond, precision: None, rounding: None, exception: false });
    }

    // A precision then a rounding mode, either of which may be absent: `ADFSP`
    // is single rounding to plus infinity, `STFP` is packed, and `FIXZ` is a
    // rounding mode with no precision at all, because a fix has no format to
    // choose. Reading the precision first is what makes `P` mean packed in
    // `STFP` and a condition in `MVFPLS`.
    let split = |r: &str| -> Option<(Option<Precision>, Option<char>)> {
        let is_round = |c: char| matches!(c, 'P' | 'M' | 'Z');
        let mut cs = r.chars().peekable();
        let precision = match cs.peek() {
            Some(c) => match Precision::from_letter(*c) {
                Some(p) => {
                    cs.next();
                    Some(p)
                }
                None if is_round(*c) => None,
                None => return None,
            },
            None => None,
        };
        let rounding = match cs.next() {
            Some(c) if is_round(c) => Some(c),
            Some(_) => return None,
            None => None,
        };
        if cs.next().is_some() {
            return None;
        }
        Some((precision, rounding))
    };

    // No condition.
    if let Some((precision, rounding)) = split(rest) {
        return Some(Fpa {
            stem,
            cond: String::new(),
            precision,
            rounding,
            exception: false,
        });
    }
    // Condition, then precision.
    if rest.len() >= 2 && CONDS.contains(&&rest[..2]) {
        if let Some((precision, rounding)) = split(&rest[2..]) {
            return Some(Fpa {
                stem,
                cond: canonical_cond(&rest[..2]).to_string(),
                precision,
                rounding,
                exception: false,
            });
        }
    }
    None
}

/// Is this mnemonic an FPA instruction?
pub fn is_fpa(mnemonic: &str) -> bool {
    parse(mnemonic).is_some()
}

/// How many instructions this becomes, for the location counter.
///
/// Layout has to agree with what the encoder is eventually handed, and a
/// compare becomes two instructions. Anything refused still occupies its one
/// word, so a routine full of extended-precision arithmetic keeps every label
/// around it at the right address.
pub fn words(mnemonic: &str) -> Option<u32> {
    let f = parse(mnemonic)?;
    Some(match f.stem {
        // VCMP leaves its result in the FPSCR; VMRS brings it to the ARM flags.
        "CMF" | "CNF" => 2,
        _ => 1,
    })
}

// ------------------------------------------------------------- registers

/// The VFP register an FPA register becomes, at this precision.
///
/// `f3` is `d3`, and `f3` in single precision is `s6` — the low half of that
/// same `d3`. One FPA register, one place to live.
fn map_register(name: &str, p: Precision) -> Option<String> {
    let n: u32 = name.trim().strip_prefix(['f', 'F'])?.parse().ok()?;
    if n > 7 {
        return None;
    }
    match p {
        Precision::Single => Some(format!("s{}", n * 2)),
        Precision::Double => Some(format!("d{n}")),
        _ => None,
    }
}

/// The double register covering an FPA register whatever its precision, for
/// the NEON immediate that zeroes it.
fn covering_double(name: &str) -> Option<String> {
    let n: u32 = name.trim().strip_prefix(['f', 'F'])?.parse().ok()?;
    (n <= 7).then(|| format!("d{n}"))
}

/// The FPA's eight immediate constants.
///
/// They are the only ones the instruction set can encode, and VFP's own
/// 8-bit floating-point immediate happens to cover all but zero.
fn map_immediate(text: &str) -> Option<String> {
    let t = text.trim().trim_start_matches('#').trim();
    if t == "0.5" {
        return Some("0.5".into());
    }
    // By the time an operand reaches here the expression evaluator has already
    // reduced it, so `#1` arrives as `#0x1`.
    let n = match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        Some(h) => u32::from_str_radix(h, 16).ok()?,
        None => t.trim_end_matches(".0").parse::<u32>().ok()?,
    };
    match n {
        0 => Some("0".into()),
        1..=5 | 10 => Some(format!("{n}.0")),
        _ => None,
    }
}

// ------------------------------------------------------------ conversion

/// What a dyadic or monadic FPA operation becomes.
///
/// `RSF` and `RDF` are the reversed forms — `Fd := Fm - Fn`, `Fd := Fm / Fn` —
/// so they become the ordinary VFP instruction with the operands swapped.
struct DataOp {
    vfp: &'static str,
    /// Reversed: the second and third operands change places.
    reversed: bool,
}

fn data_op(stem: &str) -> Option<DataOp> {
    let (vfp, reversed) = match stem {
        "ADF" => ("VADD", false),
        "SUF" => ("VSUB", false),
        "RSF" => ("VSUB", true),
        "MUF" | "FML" => ("VMUL", false),
        "DVF" | "FDV" => ("VDIV", false),
        "RDF" | "FRD" => ("VDIV", true),
        _ => return None,
    };
    Some(DataOp { vfp, reversed })
}

fn monadic_op(stem: &str) -> Option<&'static str> {
    match stem {
        "MVF" => Some("VMOV"),
        "MNF" => Some("VNEG"),
        "ABS" => Some("VABS"),
        "SQT" => Some("VSQRT"),
        _ => None,
    }
}

fn refuse(what: &str, why: &str) -> Legalized {
    Legalized::Unsupported(format!("{what}: {why}"))
}

/// Translate one FPA instruction, or say why it cannot be.
///
/// Returns `None` if this is not an FPA instruction at all.
pub fn convert(mnemonic: &str, operands: &str) -> Option<Legalized> {
    let f = parse(mnemonic)?;
    let up = mnemonic.to_ascii_uppercase();

    if TRANSCENDENTAL.contains(&f.stem) {
        return Some(refuse(
            &up,
            "an FPA transcendental; VFP has no such instruction, so this needs \
             a library call rather than a translation",
        ));
    }
    if matches!(f.stem, "LFM" | "SFM") {
        return Some(refuse(
            &up,
            "transfers registers in the FPA's 12-byte internal format, which \
             VFP has no equivalent of",
        ));
    }
    if matches!(f.stem, "FLT" | "FIX" | "URD" | "NRM" | "RMF" | "RND") {
        return Some(refuse(&up, "has no single-instruction VFP equivalent"));
    }

    // --- the status and control words --------------------------------------
    if STATUS.contains(&f.stem) {
        let rd = parts_first(operands);
        let cond = &f.cond;
        return Some(match f.stem {
            // The FPA's status word and VFP's FPSCR agree where this is used:
            // bits 0-4 are the cumulative exception flags in the same order,
            // IVO/DVZ/OFL/UFL/INX against IOC/DZC/OFC/UFC/IXC, and bits 16-20
            // are their trap enables. They disagree above that -- the FPA put
            // a system ID in the top byte where VFP puts N, Z, C and V -- so
            // this is right for the flag manipulation the sources do with it
            // and wrong for anything that reads the whole word as an FPA one.
            "RFS" => Legalized::One(format!("VMRS{cond}"), format!("{rd}, fpscr")),
            "WFS" => Legalized::One(format!("VMSR{cond}"), format!("fpscr, {rd}")),
            // The control word was the FPA's own privileged state, describing
            // hardware VFP does not have.
            _ => refuse(
                &up,
                "reads or writes the FPA's control word, which describes \
                 hardware VFP does not have",
            ),
        });
    }
    match f.precision {
        Some(Precision::Extended) => {
            return Some(refuse(
                &up,
                "is extended precision: about 64 bits of mantissa against VFP's \
                 53, so translating it would quietly lose the accuracy the code \
                 exists to keep",
            ))
        }
        Some(Precision::Packed) => {
            return Some(refuse(&up, "is packed decimal, which VFP has no format for"))
        }
        _ => {}
    }

    // Tested after the precision, so a mnemonic that is both extended and
    // rounded is reported as the extended one -- the more fundamental fact,
    // and the one that groups it with the rest of the same problem.
    if f.rounding.is_some() {
        return Some(refuse(
            &up,
            "names a rounding mode; VFP takes its rounding from the FPSCR, not \
             from the instruction",
        ));
    }

    let cond = f.cond.clone();
    let parts = crate::layout::split_top_level(operands);
    let arg = |i: usize| parts.get(i).map(|s| s.trim().to_string()).unwrap_or_default();

    // --- comparisons: VCMP then bring the flags across ---------------------
    if matches!(f.stem, "CMF" | "CNF") {
        if f.stem == "CNF" {
            return Some(refuse(
                &up,
                "compares against the negation of its operand, which VFP would \
                 need a separate VNEG for",
            ));
        }
        // The operands carry no precision of their own, so the registers are
        // compared as the doubles they live in.
        let (Some(a), b) = (covering_double(&arg(0)), arg(1)) else {
            return Some(refuse(&up, "operands are not FPA registers"));
        };
        let rhs = match covering_double(&b) {
            Some(r) => r,
            None => match map_immediate(&b) {
                // VCMP against zero is the only immediate it takes.
                Some(v) if v == "0" => "#0".to_string(),
                _ => return Some(refuse(&up, "VCMP compares registers, or against zero")),
            },
        };
        let op = if f.exception { "VCMPE" } else { "VCMP" };
        return Some(Legalized::Many(vec![
            (format!("{op}{cond}.F64"), format!("{a}, {rhs}")),
            (format!("VMRS{cond}"), "APSR_nzcv, fpscr".to_string()),
        ]));
    }

    // Everything past here needs a precision VFP has.
    let Some(p) = f.precision else {
        return Some(refuse(&up, "carries no precision suffix"));
    };
    let Some(ty) = p.suffix() else {
        return Some(refuse(&up, "is not a precision VFP has"));
    };

    // --- loads and stores --------------------------------------------------
    if matches!(f.stem, "LDF" | "STF") {
        let Some(rd) = map_register(&arg(0), p) else {
            return Some(refuse(&up, "first operand is not an FPA register"));
        };
        let addr = operands
            .split_once(',')
            .map(|(_, rest)| rest.trim().to_string())
            .unwrap_or_default();
        return Some(transfer(f.stem == "LDF", &cond, &rd, &addr, p.width(), &up));
    }

    // --- monadic: a destination and one source or an immediate -------------
    if let Some(op) = monadic_op(f.stem) {
        let Some(rd) = map_register(&arg(0), p) else {
            return Some(refuse(&up, "destination is not an FPA register"));
        };
        let src = arg(1);
        if let Some(rm) = map_register(&src, p) {
            return Some(Legalized::One(
                format!("{op}{cond}.{ty}"),
                format!("{rd}, {rm}"),
            ));
        }
        // `MVF Fd, #1` is the only immediate form.
        if op == "VMOV" {
            let Some(v) = map_immediate(&src) else {
                return Some(refuse(&up, "not one of the FPA's eight immediates"));
            };
            if v == "0" {
                // VFP's floating-point immediate cannot encode zero, but the
                // Advanced SIMD one can, and clearing the whole `d` register
                // is safe: the odd `s` registers are unused by this mapping.
                let Some(dd) = covering_double(&arg(0)) else {
                    return Some(refuse(&up, "destination is not an FPA register"));
                };
                let width = if p == Precision::Single { "I32" } else { "I64" };
                return Some(Legalized::One(
                    format!("VMOV{cond}.{width}"),
                    format!("{dd}, #0"),
                ));
            }
            return Some(Legalized::One(
                format!("VMOV{cond}.{ty}"),
                format!("{rd}, #{v}"),
            ));
        }
        return Some(refuse(&up, "source is not an FPA register"));
    }

    // --- dyadic ------------------------------------------------------------
    if let Some(d) = data_op(f.stem) {
        let (Some(rd), Some(rn)) = (map_register(&arg(0), p), map_register(&arg(1), p)) else {
            return Some(refuse(&up, "operands are not FPA registers"));
        };
        let third = arg(2);
        let Some(rm) = map_register(&third, p) else {
            // The FPA allows an immediate here; VFP does not.
            return Some(refuse(
                &up,
                "VFP arithmetic takes no immediate operand, so this needs the \
                 constant in a register first",
            ));
        };
        let (a, b) = if d.reversed { (rm, rn) } else { (rn, rm) };
        return Some(Legalized::One(
            format!("{}{cond}.{ty}", d.vfp),
            format!("{rd}, {a}, {b}"),
        ));
    }

    Some(refuse(&up, "not translated"))
}

/// A single load or store, given an FPA addressing mode.
///
/// `VLDR` and `VSTR` take an offset and nothing else — no writeback, no
/// post-indexing — so the indexed forms become `VLDM`/`VSTM`, which do have
/// writeback. That covers the stack idioms exactly, because the FPA ones
/// always step by the transfer's own width.
fn transfer(load: bool, cond: &str, rd: &str, addr: &str, width: u32, up: &str) -> Legalized {
    let a = addr.trim();
    let plain = format!("{}{cond}", if load { "VLDR" } else { "VSTR" });

    // `[Rn], #n` -- post-indexed.
    if let Some((base, off)) = a.split_once("],") {
        let base = base.trim_start_matches('[').trim();
        let n = parse_offset(off);
        if n == Some(width as i64) {
            let op = if load { "VLDMIA" } else { "VSTMIA" };
            return Legalized::One(format!("{op}{cond}"), format!("{base}!, {{{rd}}}"));
        }
        return Legalized::Unsupported(format!(
            "{up}: post-indexed by {} where VFP can only step by the transfer's \
             own {width} bytes",
            off.trim()
        ));
    }

    // `[Rn, #n]!` -- pre-indexed with writeback.
    if let Some(inner) = a.strip_suffix("]!") {
        let inner = inner.trim_start_matches('[');
        let (base, off) = inner.split_once(',').unwrap_or((inner, "#0"));
        let n = parse_offset(off);
        if n == Some(-(width as i64)) {
            let op = if load { "VLDMDB" } else { "VSTMDB" };
            return Legalized::One(
                format!("{op}{cond}"),
                format!("{}!, {{{rd}}}", base.trim()),
            );
        }
        return Legalized::Unsupported(format!(
            "{up}: pre-indexed by {} where VFP can only step by the transfer's \
             own {width} bytes",
            off.trim()
        ));
    }

    // `[Rn, #n]` or `[Rn]` or a label: VLDR takes these unchanged.
    Legalized::One(plain, format!("{rd}, {a}"))
}

/// The first operand, trimmed.
fn parts_first(operands: &str) -> String {
    operands
        .split(',')
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

/// The signed byte offset an addressing operand carries, if it is a constant.
fn parse_offset(text: &str) -> Option<i64> {
    let t = text.trim().trim_end_matches([']', '!']).trim();
    let t = t.trim_start_matches('#').trim();
    let (neg, digits) = match t.strip_prefix('-') {
        Some(d) => (true, d.trim()),
        None => (false, t),
    };
    let v = match digits.strip_prefix("0x").or_else(|| digits.strip_prefix("0X")) {
        Some(h) => i64::from_str_radix(h, 16).ok()?,
        None => digits.parse::<i64>().ok()?,
    };
    Some(if neg { -v } else { v })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(m: &str, o: &str) -> (String, String) {
        match convert(m, o) {
            Some(Legalized::One(a, b)) => (a, b),
            other => panic!("{m} {o}: expected one instruction, got {other:?}"),
        }
    }

    fn why(m: &str, o: &str) -> String {
        match convert(m, o) {
            Some(Legalized::Unsupported(w)) => w,
            other => panic!("{m} {o}: expected a refusal, got {other:?}"),
        }
    }

    // ---- parsing --------------------------------------------------------

    #[test]
    fn a_plain_mnemonic_parses() {
        let f = parse("LDFD").unwrap();
        assert_eq!(f.stem, "LDF");
        assert_eq!(f.precision, Some(Precision::Double));
        assert!(f.cond.is_empty());
    }

    #[test]
    fn the_condition_comes_before_the_precision() {
        // `MVFNEE` is MVF, conditional on NE, extended.
        let f = parse("MVFNEE").unwrap();
        assert_eq!((f.stem, f.cond.as_str()), ("MVF", "NE"));
        assert_eq!(f.precision, Some(Precision::Extended));
    }

    #[test]
    fn pl_is_a_condition_not_a_packed_rounding_mode() {
        // The one ambiguous shape: `MVFPLS` cannot be packed then `LS`,
        // because `LS` is not a rounding mode.
        let f = parse("MVFPLS").unwrap();
        assert_eq!((f.stem, f.cond.as_str()), ("MVF", "PL"));
        assert_eq!(f.precision, Some(Precision::Single));
        // Whereas a bare P really is packed.
        assert_eq!(parse("STFP").unwrap().precision, Some(Precision::Packed));
    }

    #[test]
    fn a_rounding_mode_follows_the_precision() {
        let f = parse("ADFSP").unwrap();
        assert_eq!(f.precision, Some(Precision::Single));
        assert_eq!(f.rounding, Some('P'));
    }

    #[test]
    fn a_comparison_takes_no_precision() {
        assert_eq!(parse("CMF").unwrap().precision, None);
        assert_eq!(parse("CMFNE").unwrap().cond, "NE");
        assert!(parse("CMFE").unwrap().exception);
    }

    #[test]
    fn an_alternate_condition_spelling_is_canonicalised() {
        assert_eq!(parse("LDFLOD").unwrap().cond, "CC");
        assert_eq!(parse("LDFHSD").unwrap().cond, "CS");
    }

    #[test]
    fn ordinary_arm_mnemonics_are_not_fpa() {
        for m in ["MOV", "LDR", "STR", "ADD", "B", "BL", "SWI", "LDMFD"] {
            assert!(!is_fpa(m), "{m} must not parse as FPA");
        }
    }

    // ---- registers ------------------------------------------------------

    #[test]
    fn an_fpa_register_has_one_place_to_live() {
        // f3 is d3, and f3 as a single is the low half of that same d3.
        assert_eq!(map_register("f3", Precision::Double).unwrap(), "d3");
        assert_eq!(map_register("f3", Precision::Single).unwrap(), "s6");
        // So no two FPA registers can collide: the odd s registers are unused.
        let mut used: Vec<String> = Vec::new();
        for n in 0..8 {
            used.push(map_register(&format!("f{n}"), Precision::Single).unwrap());
            used.push(map_register(&format!("f{n}"), Precision::Double).unwrap());
        }
        assert_eq!(used.len(), 16);
        // s2n lies inside dn and nowhere else.
        assert_eq!(map_register("f0", Precision::Single).unwrap(), "s0");
        assert_eq!(map_register("f1", Precision::Single).unwrap(), "s2");
    }

    #[test]
    fn there_are_only_eight_fpa_registers() {
        assert!(map_register("f8", Precision::Double).is_none());
        assert!(map_register("r0", Precision::Double).is_none());
    }

    // ---- data operations ------------------------------------------------

    #[test]
    fn arithmetic_maps_by_precision() {
        assert_eq!(one("ADFD", "f0, f1, f2"), ("VADD.F64".into(), "d0, d1, d2".into()));
        assert_eq!(one("ADFS", "f0, f1, f2"), ("VADD.F32".into(), "s0, s2, s4".into()));
        assert_eq!(one("MUFD", "f0, f0, f1"), ("VMUL.F64".into(), "d0, d0, d1".into()));
        assert_eq!(one("DVFS", "f1, f2, f3"), ("VDIV.F32".into(), "s2, s4, s6".into()));
    }

    #[test]
    fn the_reversed_forms_swap_their_operands() {
        // RSF is Fd := Fm - Fn, so VSUB takes them the other way round.
        assert_eq!(one("RSFD", "f0, f1, f2"), ("VSUB.F64".into(), "d0, d2, d1".into()));
        assert_eq!(one("RDFD", "f0, f1, f2"), ("VDIV.F64".into(), "d0, d2, d1".into()));
        // And the forward ones do not.
        assert_eq!(one("SUFD", "f0, f1, f2"), ("VSUB.F64".into(), "d0, d1, d2".into()));
    }

    #[test]
    fn moves_negations_and_roots() {
        assert_eq!(one("MVFD", "f0, f1"), ("VMOV.F64".into(), "d0, d1".into()));
        assert_eq!(one("MNFS", "f0, f1"), ("VNEG.F32".into(), "s0, s2".into()));
        assert_eq!(one("ABSD", "f0, f0"), ("VABS.F64".into(), "d0, d0".into()));
        assert_eq!(one("SQTD", "f0, f1"), ("VSQRT.F64".into(), "d0, d1".into()));
    }

    #[test]
    fn a_condition_carries_onto_the_vfp_instruction() {
        assert_eq!(one("MVFMID", "f0, f1"), ("VMOVMI.F64".into(), "d0, d1".into()));
        assert_eq!(one("ADFNES", "f0, f1, f2"), ("VADDNE.F32".into(), "s0, s2, s4".into()));
    }

    #[test]
    fn the_fpa_immediates_map_except_zero() {
        assert_eq!(one("MVFD", "f0, #1"), ("VMOV.F64".into(), "d0, #1.0".into()));
        assert_eq!(one("MVFS", "f0, #0.5"), ("VMOV.F32".into(), "s0, #0.5".into()));
        // VFP's floating-point immediate cannot encode zero; the SIMD one can,
        // and clearing the whole d register is safe under this mapping.
        assert_eq!(one("MVFD", "f2, #0"), ("VMOV.I64".into(), "d2, #0".into()));
        assert_eq!(one("MVFS", "f2, #0"), ("VMOV.I32".into(), "d2, #0".into()));
        // Anything outside the FPA's eight is not an FPA immediate at all.
        assert!(why("MVFD", "f0, #7").contains("eight immediates"));
        // And they arrive already evaluated, in hex.
        assert_eq!(one("MVFD", "f0, #0x1"), ("VMOV.F64".into(), "d0, #1.0".into()));
        assert_eq!(one("MVFD", "f0, #0xA"), ("VMOV.F64".into(), "d0, #10.0".into()));
    }

    // ---- transfers ------------------------------------------------------

    #[test]
    fn a_plain_offset_load_is_a_vldr() {
        assert_eq!(one("LDFD", "f0, [sp, #8]"), ("VLDR".into(), "d0, [sp, #8]".into()));
        assert_eq!(one("STFS", "f1, [r4]"), ("VSTR".into(), "s2, [r4]".into()));
        // A label works the same way.
        assert_eq!(one("LDFD", "f0, .+136"), ("VLDR".into(), "d0, .+136".into()));
    }

    #[test]
    fn the_stack_idioms_become_writeback_transfers() {
        // VLDR has no writeback, but VLDM does, and the FPA forms always step
        // by exactly the transfer's width.
        assert_eq!(one("LDFD", "f0, [sp], #8"), ("VLDMIA".into(), "sp!, {d0}".into()));
        assert_eq!(one("STFD", "f0, [sp, #-8]!"), ("VSTMDB".into(), "sp!, {d0}".into()));
        assert_eq!(one("LDFS", "f0, [sp], #4"), ("VLDMIA".into(), "sp!, {s0}".into()));
        assert_eq!(one("STFS", "f1, [r4, #-4]!"), ("VSTMDB".into(), "r4!, {s2}".into()));
    }

    #[test]
    fn an_indexed_step_of_the_wrong_size_is_refused() {
        // Nothing in VFP steps a base register by an arbitrary amount.
        assert!(why("LDFD", "f0, [sp], #16").contains("post-indexed"));
        assert!(why("STFD", "f0, [sp, #-24]!").contains("pre-indexed"));
    }

    // ---- comparisons ----------------------------------------------------

    #[test]
    fn a_compare_becomes_two_instructions() {
        // The FPA set the ARM flags; VFP sets the FPSCR and VMRS copies across.
        let Some(Legalized::Many(v)) = convert("CMF", "f0, f1") else {
            panic!("expected an expansion")
        };
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], ("VCMP.F64".into(), "d0, d1".into()));
        assert_eq!(v[1], ("VMRS".into(), "APSR_nzcv, fpscr".into()));
        // And layout has to know that before the encoder runs.
        assert_eq!(words("CMF"), Some(2));
        assert_eq!(words("LDFD"), Some(1));
    }

    #[test]
    fn a_conditional_compare_carries_its_condition_to_both_halves() {
        let Some(Legalized::Many(v)) = convert("CMFNE", "f0, f1") else { panic!() };
        assert_eq!(v[0].0, "VCMPNE.F64");
        assert_eq!(v[1].0, "VMRSNE", "the flag move must be conditional too");
    }

    #[test]
    fn the_exception_raising_compare_uses_vcmpe() {
        let Some(Legalized::Many(v)) = convert("CMFE", "f0, f1") else { panic!() };
        assert_eq!(v[0].0, "VCMPE.F64");
    }

    #[test]
    fn a_compare_against_zero_is_the_one_immediate_vcmp_takes() {
        let Some(Legalized::Many(v)) = convert("CMF", "f0, #0") else { panic!() };
        assert_eq!(v[0], ("VCMP.F64".into(), "d0, #0".into()));
        assert!(why("CMF", "f0, #1").contains("or against zero"));
    }

    // ---- what cannot be translated --------------------------------------

    #[test]
    fn extended_precision_is_refused_and_says_why() {
        for m in ["LDFE", "STFE", "ADFE", "MUFE", "MVFE", "SQTE"] {
            let w = why(m, "f0, f1, f2");
            assert!(w.contains("extended precision"), "{m}: {w}");
            assert!(w.contains("53"), "{m} must say what is lost: {w}");
        }
    }

    #[test]
    fn packed_decimal_is_refused() {
        assert!(why("STFP", "f0, [sp]").contains("packed decimal"));
    }

    #[test]
    fn the_transcendentals_are_refused_as_library_calls() {
        for m in ["SIND", "COSD", "LOGD", "LGNE", "EXPD", "POWVCD", "ATND"] {
            let w = why(m, "f0, f1");
            assert!(w.contains("library call"), "{m}: {w}");
        }
    }

    #[test]
    fn multiple_register_transfers_are_refused() {
        for m in ["LFMFD", "SFMFD", "LFM", "SFM"] {
            assert!(why(m, "f0, 4, [sp]").contains("12-byte"), "{m}");
        }
    }

    #[test]
    fn a_named_rounding_mode_is_refused() {
        let w = why("ADFSZ", "f0, f1, f2");
        assert!(w.contains("rounding mode"), "{w}");
        assert!(w.contains("FPSCR"), "{w}");
    }

    #[test]
    fn arithmetic_with_an_immediate_is_refused() {
        // The FPA takes one; VFP does not.
        assert!(why("ADFD", "f0, f1, #1").contains("no immediate operand"));
    }

    #[test]
    fn a_negated_compare_is_refused() {
        assert!(why("CNF", "f0, f1").contains("negation"));
    }

    #[test]
    fn everything_refused_still_occupies_its_word() {
        // Otherwise every label after a refused instruction would move.
        for m in ["LDFE", "STFP", "SIND", "LFMFD", "FLTD", "FIXZ"] {
            assert_eq!(words(m), Some(1), "{m}");
        }
    }

    #[test]
    fn the_status_word_moves_through_the_fpscr() {
        assert_eq!(one("RFS", "r2"), ("VMRS".into(), "r2, fpscr".into()));
        assert_eq!(one("WFS", "r3"), ("VMSR".into(), "fpscr, r3".into()));
        assert_eq!(one("RFSNE", "r0"), ("VMRSNE".into(), "r0, fpscr".into()));
        assert_eq!(words("RFS"), Some(1));
    }

    #[test]
    fn the_control_word_is_refused() {
        // It described the FPA's own hardware, which is not there.
        assert!(why("RFC", "r0").contains("control word"));
        assert!(why("WFC", "r0").contains("control word"));
    }

    #[test]
    fn a_non_fpa_mnemonic_converts_to_nothing() {
        assert!(convert("MOV", "r0, #1").is_none());
        assert!(words("MOV").is_none());
    }
}
