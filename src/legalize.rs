//! Legalization: Acorn instruction forms to something LLVM will encode.
//!
//! The pipeline runs
//!
//! ```text
//!   ObjAsm source -> expand -> lower (syntax) -> LEGALIZE -> LLVM -> AOF
//! ```
//!
//! `lower` is textual: it renames `SWI` to `SVC`, reorders a pre-UAL suffix,
//! substitutes register aliases and rewrites number literals. Legalization is
//! the stage that decides what an instruction can *become*, because some Acorn
//! forms have no UAL spelling at all:
//!
//! * **`ADRL`** is a pseudo-instruction. ObjAsm expands it into two
//!   data-processing instructions so it reaches further than `ADR`'s single
//!   rotated immediate allows. LLVM has never heard of it, so we expand it
//!   ourselves -- which is possible because after the first pass we know both
//!   the location counter and the target.
//! * **`TEQP`, `TSTP`, `CMPP`, `CMNP`** wrote the PSR directly on a 26-bit
//!   ARM. There is no 32-bit equivalent and the encoder rejects them.
//! * **`SWP`** is deprecated: "requires armv7 or earlier".
//!
//! Anything with no legal form becomes a raw word, so one unencodable
//! instruction costs one instruction rather than the whole file.

use crate::lower;

/// What an Acorn instruction legalizes to.
#[derive(Debug, Clone, PartialEq)]
pub enum Legalized {
    /// A single UAL instruction the encoder will accept.
    One(String, String),
    /// A pseudo-instruction that expands to several real ones.
    Many(Vec<(String, String)>),
    /// No UAL form exists; emit this word directly with `.inst`.
    RawWord(u32),
    /// Cannot be legalized here, with the reason.
    Unsupported(String),
}

/// What legalization needs to know about where it is.
pub struct Context {
    /// Address of the instruction being legalized, within its area.
    pub here: u32,
    /// Value of the operand's target, when it is known.
    pub target: Option<u32>,
}

// ------------------------------------------------------- ARM immediates

/// Can this value be an ARM data-processing immediate?
///
/// The field is eight bits rotated right by an even amount, so only a small
/// set of values fits -- which is the entire reason `ADRL` exists.
pub fn as_arm_immediate(v: u32) -> Option<(u32, u32)> {
    if v <= 0xFF {
        return Some((0, v));
    }
    for rot in 1..16u32 {
        let shifted = v.rotate_left(2 * rot);
        if shifted <= 0xFF {
            return Some((rot, shifted));
        }
    }
    None
}

/// Split a value into a sum of ARM immediates, smallest number first.
///
/// Takes the lowest set bit each time and consumes the eight bits above it,
/// which is what ObjAsm does when expanding `ADRL`. Returns `None` if more
/// parts would be needed than allowed.
pub fn split_immediates(mut v: u32, max_parts: usize) -> Option<Vec<u32>> {
    if v == 0 {
        return Some(vec![0]);
    }
    let mut parts = Vec::new();
    while v != 0 {
        if parts.len() == max_parts {
            return None;
        }
        // Round the lowest set bit down to an even position, since the
        // rotation is by an even number of bits.
        let low = v.trailing_zeros() & !1;
        let part = v & (0xFF << low);
        if part == 0 {
            return None;
        }
        parts.push(part);
        v &= !part;
    }
    Some(parts)
}

/// Expand `ADRL Rd, target` into the instructions ObjAsm would generate.
///
/// `pc` reads as the instruction's own address plus eight, so the offset is
/// measured from there. A negative offset becomes `SUB`.
pub fn expand_adrl(cond: &str, rd: &str, here: u32, target: u32) -> Legalized {
    let delta = (target as i64) - (here as i64 + 8);
    let (op, mag) = if delta >= 0 {
        ("ADD", delta as u32)
    } else {
        ("SUB", (-delta) as u32)
    };
    let Some(parts) = split_immediates(mag, 2) else {
        return Legalized::Unsupported(format!(
            "ADRL offset {delta} needs more than two instructions"
        ));
    };
    let mut out = Vec::new();
    // The first instruction works from pc, the rest from the destination.
    for (i, p) in parts.iter().enumerate() {
        let src = if i == 0 { "pc" } else { rd };
        out.push((format!("{op}{cond}"), format!("{rd}, {src}, #{p}")));
    }
    // A single part still needs two instructions to keep the size fixed:
    // ObjAsm's ADRL always occupies eight bytes, and the location counter
    // has already been advanced on that basis.
    if out.len() == 1 {
        out.push((format!("{op}{cond}"), format!("{rd}, {rd}, #0")));
    }
    Legalized::Many(out)
}

/// Expand `ADR Rd, target` into the single instruction it stands for.
///
/// Unlike `ADRL` this is one instruction, so the offset has to fit in a single
/// rotated immediate; if it does not, the source wanted `ADRL`.
pub fn expand_adr(cond: &str, rd: &str, here: u32, target: u32) -> Legalized {
    let delta = (target as i64) - (here as i64 + 8);
    let (op, mag) = if delta >= 0 {
        ("ADD", delta as u32)
    } else {
        ("SUB", (-delta) as u32)
    };
    if as_arm_immediate(mag).is_none() {
        return Legalized::Unsupported(format!(
            "ADR offset {delta} does not fit one instruction; ADRL reaches further"
        ));
    }
    Legalized::One(format!("{op}{cond}"), format!("{rd}, pc, #{mag}"))
}

/// A literal already reduced to a number by the expression evaluator.
fn parse_number(s: &str) -> Result<u32, ()> {
    let s = s.trim();
    let v = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => u32::from_str_radix(h, 16),
        None => s.parse::<u32>(),
    };
    v.map_err(|_| ())
}

/// The condition suffix on a mnemonic whose stem is known.
fn condition_of(mnemonic: &str, stem: &str) -> String {
    mnemonic
        .to_ascii_uppercase()
        .strip_prefix(stem)
        .unwrap_or("")
        .to_string()
}

/// Legalize one instruction.
pub fn legalize(mnemonic: &str, operands: &str, ctx: &Context) -> Legalized {
    let up = mnemonic.to_ascii_uppercase();

    if lower::is_adr(&up) {
        let cond = condition_of(&up, "ADR");
        let rd = operands.split(',').next().unwrap_or("r0").trim().to_string();
        return match ctx.target {
            Some(t) => expand_adr(&cond, &rd, ctx.here, t),
            None => Legalized::Unsupported("ADR to an unresolved target".into()),
        };
    }

    if lower::is_adrl(&up) {
        let cond = condition_of(&up, "ADRL");
        let rd = operands.split(',').next().unwrap_or("r0").trim().to_string();
        return match ctx.target {
            Some(t) => expand_adrl(&cond, &rd, ctx.here, t),
            None => Legalized::Unsupported("ADRL to an unresolved target".into()),
        };
    }

    // `LDR Rd, =value` asks for the value in a register by whatever means.
    // ObjAsm counts it as one instruction, so it has to stay one: a value that
    // fits an immediate becomes MOV, its complement MVN. Anything else needs a
    // literal pool, which is LTORG's business and not yet built.
    if up.starts_with("LDR") {
        if let Some(rest) = operands.split_once('=') {
            let cond = condition_of(&up, "LDR");
            let rd = rest.0.trim_end_matches([',', ' ']).trim().to_string();
            if let Ok(v) = parse_number(rest.1.trim()) {
                if as_arm_immediate(v).is_some() {
                    return Legalized::One(format!("MOV{cond}"), format!("{rd}, #{v}"));
                }
                if as_arm_immediate(!v).is_some() {
                    return Legalized::One(format!("MVN{cond}"), format!("{rd}, #{}", !v));
                }
                return Legalized::Unsupported(format!(
                    "LDR {rd},=&{v:X} needs a literal pool"
                ));
            }
            return Legalized::Unsupported(format!("LDR {rd},={} is unresolved", rest.1.trim()));
        }
    }

    if lower::is_psr_form(&up) {
        return Legalized::Unsupported(format!(
            "{up} writes the PSR in 26-bit mode and has no 32-bit form"
        ));
    }

    if up.starts_with("SWP") {
        return Legalized::Unsupported(format!("{up} is deprecated beyond ARMv7"));
    }

    let m = lower::normalise_mnemonic(&up).unwrap_or(up);
    Legalized::One(m, operands.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_values_are_immediates_without_rotation() {
        assert_eq!(as_arm_immediate(0), Some((0, 0)));
        assert_eq!(as_arm_immediate(0xFF), Some((0, 0xFF)));
    }

    #[test]
    fn a_rotated_byte_is_an_immediate() {
        // 0x3F00 is 0xFC rotated; 0xFF000000 is 0xFF rotated by 8.
        assert!(as_arm_immediate(0xFF00).is_some());
        assert!(as_arm_immediate(0xFF00_0000).is_some());
    }

    #[test]
    fn a_value_spanning_more_than_eight_bits_is_not_an_immediate() {
        // This is exactly the case ADRL exists to handle.
        assert_eq!(as_arm_immediate(0x1234), None);
        assert_eq!(as_arm_immediate(0x101), None);
    }

    #[test]
    fn splitting_reproduces_the_value() {
        for v in [0x1234u32, 0x101, 0xABCD, 0xFFFF, 0x12345] {
            let parts = split_immediates(v, 4).unwrap_or_else(|| panic!("{v:#x}"));
            assert_eq!(parts.iter().sum::<u32>(), v, "{v:#x} must be reconstructed");
            for p in &parts {
                assert!(as_arm_immediate(*p).is_some(), "{p:#x} must be encodable");
            }
        }
    }

    #[test]
    fn a_value_needing_too_many_parts_is_refused() {
        // Alternating bits cannot be covered by two 8-bit windows.
        assert_eq!(split_immediates(0x5555_5555, 2), None);
    }

    #[test]
    fn adrl_expands_to_two_instructions_from_pc() {
        // pc reads as the instruction's address + 8.
        let l = expand_adrl("", "r0", 0x1000, 0x1000 + 8 + 0x1234);
        let Legalized::Many(v) = l else { panic!("expected an expansion") };
        assert_eq!(v.len(), 2, "ADRL always occupies eight bytes");
        assert_eq!(v[0].0, "ADD");
        assert!(v[0].1.starts_with("r0, pc, #"), "first works from pc: {}", v[0].1);
        assert!(v[1].1.starts_with("r0, r0, #"), "second from the destination");
        // The parts must add up to the offset.
        let sum: u32 = v
            .iter()
            .map(|(_, o)| o.rsplit('#').next().unwrap().parse::<u32>().unwrap())
            .sum();
        assert_eq!(sum, 0x1234);
    }

    #[test]
    fn a_backward_adrl_subtracts() {
        let l = expand_adrl("", "r1", 0x2000, 0x1000);
        let Legalized::Many(v) = l else { panic!("expected an expansion") };
        assert!(v.iter().all(|(m, _)| m == "SUB"), "backwards means SUB");
        let sum: u32 = v
            .iter()
            .map(|(_, o)| o.rsplit('#').next().unwrap().parse::<u32>().unwrap())
            .sum();
        assert_eq!(sum, 0x2000 + 8 - 0x1000);
    }

    #[test]
    fn a_conditional_adrl_keeps_its_condition() {
        let l = expand_adrl("EQ", "r2", 0, 0x108);
        let Legalized::Many(v) = l else { panic!() };
        assert!(v.iter().all(|(m, _)| m == "ADDEQ"), "{v:?}");
    }

    #[test]
    fn adrl_still_takes_eight_bytes_when_one_would_do() {
        // The location counter was advanced by eight during expansion, so the
        // expansion must occupy eight bytes even for a short offset.
        let l = expand_adrl("", "r0", 0, 8 + 4);
        let Legalized::Many(v) = l else { panic!() };
        assert_eq!(v.len(), 2);
        assert!(v[1].1.ends_with("#0"), "padded with a no-op add");
    }

    #[test]
    fn adr_is_one_instruction_measured_from_pc() {
        // pc reads as the instruction's address + 8, so from &04 a label at
        // &24 is 24 bytes away, not 32.
        assert_eq!(
            expand_adr("", "r0", 0x04, 0x24),
            Legalized::One("ADD".into(), "r0, pc, #24".into())
        );
    }

    #[test]
    fn a_backward_adr_subtracts() {
        assert_eq!(
            expand_adr("NE", "r1", 0x40, 0x10),
            Legalized::One("SUBNE".into(), "r1, pc, #56".into())
        );
    }

    #[test]
    fn an_adr_too_far_for_one_immediate_is_refused() {
        // This is what ADRL is for, and the message says so.
        match expand_adr("", "r0", 0, 0x1234 + 8) {
            Legalized::Unsupported(why) => assert!(why.contains("ADRL"), "{why}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn an_adr_with_no_known_target_is_reported() {
        let ctx = Context { here: 0, target: None };
        assert!(matches!(
            legalize("ADR", "r0, Somewhere", &ctx),
            Legalized::Unsupported(_)
        ));
    }

    #[test]
    fn an_adrl_with_no_known_target_is_reported() {
        let ctx = Context { here: 0, target: None };
        let l = legalize("ADRL", "r0, Somewhere", &ctx);
        assert!(matches!(l, Legalized::Unsupported(_)));
    }

    #[test]
    fn an_ldr_of_a_small_literal_becomes_a_move() {
        let ctx = Context { here: 0, target: None };
        assert_eq!(
            legalize("LDR", "r0, =0x10", &ctx),
            Legalized::One("MOV".into(), "r0, #16".into())
        );
        // A condition carries across.
        assert_eq!(
            legalize("LDREQ", "r1, =0xFF", &ctx),
            Legalized::One("MOVEQ".into(), "r1, #255".into())
        );
    }

    #[test]
    fn an_ldr_of_a_complemented_literal_becomes_mvn() {
        let ctx = Context { here: 0, target: None };
        // -1 is not an immediate; its complement, 0, is.
        assert_eq!(
            legalize("LDR", "r0, =0xFFFFFFFF", &ctx),
            Legalized::One("MVN".into(), "r0, #0".into())
        );
    }

    #[test]
    fn an_ldr_needing_a_pool_says_so() {
        let ctx = Context { here: 0, target: None };
        match legalize("LDR", "r0, =0x12345678", &ctx) {
            Legalized::Unsupported(why) => assert!(why.contains("literal pool"), "{why}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn the_26_bit_psr_forms_are_unsupported() {
        let ctx = Context { here: 0, target: None };
        for m in ["TEQP", "TSTP", "CMPP", "CMNP"] {
            match legalize(m, "r0, #1", &ctx) {
                Legalized::Unsupported(why) => assert!(why.contains("26-bit"), "{why}"),
                other => panic!("{m} should be unsupported, got {other:?}"),
            }
        }
    }

    #[test]
    fn swp_is_reported_as_deprecated() {
        let ctx = Context { here: 0, target: None };
        assert!(matches!(
            legalize("SWP", "r0, r1, [r2]", &ctx),
            Legalized::Unsupported(_)
        ));
    }

    #[test]
    fn an_ordinary_instruction_passes_through_normalised() {
        let ctx = Context { here: 0, target: None };
        assert_eq!(
            legalize("SUBNES", "r1, r1, #1", &ctx),
            Legalized::One("SUBSNE".into(), "r1, r1, #1".into())
        );
        assert_eq!(
            legalize("SWI", "&10", &ctx),
            Legalized::One("SVC".into(), "&10".into())
        );
        assert_eq!(
            legalize("CMPS", "r0, #1", &ctx),
            Legalized::One("CMP".into(), "r0, #1".into())
        );
    }
}
