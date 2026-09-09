//! `aofdump` — show what is inside an AOF object.
//!
//!     aofdump <object> [-x]
//!
//! Roughly what the DDE's `decaof` prints, in the same order, so the two can
//! be read side by side. `-x` disassembles nothing; it dumps each area's words
//! with their offsets, which is enough to compare encodings by eye.

use rosasm::aof::{self, area_attr, sym_attr, FieldType, RelocBy};

/// Attribute bits, named. Code and data reuse two bit positions, so which set
/// applies depends on the CODE bit.
fn describe_area(attr: u32) -> String {
    let mut v: Vec<&str> = Vec::new();
    let code = attr & area_attr::CODE != 0;
    for (bit, name) in [
        (area_attr::ABSOLUTE, "ABS"),
        (area_attr::CODE, "CODE"),
        (area_attr::COMMON_DEF, "COMDEF"),
        (area_attr::COMMON_REF, "COMMON"),
        (area_attr::ZERO_INIT, "NOINIT"),
        (area_attr::READ_ONLY, "READONLY"),
        (area_attr::POSITION_INDEPENDENT, "PIC"),
        (area_attr::DEBUG_TABLES, "DEBUG"),
    ] {
        if attr & bit != 0 {
            v.push(name);
        }
    }
    if code {
        for (bit, name) in [
            (area_attr::APCS_32, "APCS-32"),
            (area_attr::REENTRANT, "REENTRANT"),
            (area_attr::EXTENDED_FP, "EXTFP"),
            (area_attr::NO_STACK_CHECK, "NOSTACKCHECK"),
            (area_attr::THUMB, "THUMB"),
            (area_attr::HALFWORD_INSTRS, "HALFWORD"),
            (area_attr::INTERWORKING, "INTERWORK"),
        ] {
            if attr & bit != 0 {
                v.push(name);
            }
        }
    } else {
        for (bit, name) in [
            (area_attr::BASED, "BASED"),
            (area_attr::SHARED_LIB_STUB, "SLSTUB"),
        ] {
            if attr & bit != 0 {
                v.push(name);
            }
        }
    }
    if v.is_empty() {
        v.push("READWRITE");
    }
    v.join(", ")
}

fn describe_symbol(attr: u32) -> String {
    let mut v: Vec<&str> = Vec::new();
    // Defined-and-global is a defining occurrence; global alone is a
    // reference the linker must satisfy elsewhere.
    v.push(if attr & sym_attr::DEFINED != 0 {
        "defined"
    } else {
        "reference"
    });
    if attr & sym_attr::GLOBAL != 0 {
        v.push("global");
    } else {
        v.push("local");
    }
    for (bit, name) in [
        (sym_attr::ABSOLUTE, "absolute"),
        (sym_attr::CASE_INSENSITIVE, "case-insensitive"),
        (sym_attr::WEAK, "weak"),
        (sym_attr::STRONG, "strong"),
        (sym_attr::COMMON, "common"),
    ] {
        if attr & bit != 0 {
            v.push(name);
        }
    }
    v.join(", ")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let hex = args.iter().any(|a| a == "-x");
    let Some(path) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: aofdump <object> [-x]");
        std::process::exit(2);
    };
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("aofdump: {path}: {e}");
            std::process::exit(1);
        }
    };
    let obj = match aof::read(&bytes) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("aofdump: {path}: {e}");
            std::process::exit(1);
        }
    };

    println!("** Object file {path}");
    println!("   produced by: {}", obj.identification);
    match obj.entry {
        Some((a, off)) => println!("   entry point: area {a}, offset &{off:X}"),
        None => println!("   entry point: none"),
    }
    println!("   {} area(s), {} symbol(s)", obj.areas.len(), obj.symbols.len());

    for (i, a) in obj.areas.iter().enumerate() {
        let size = (a.data.len() as u32).max(a.reserved);
        println!();
        println!("** Area {i}: {}", a.name);
        println!(
            "   size &{size:X} ({size}), alignment 2^{}, attributes: {}",
            a.alignment,
            describe_area(a.attributes)
        );
        if a.attributes & area_attr::ABSOLUTE != 0 {
            println!("   base &{:08X}", a.base);
        }
        if hex && !a.data.is_empty() {
            for (n, w) in a.data.chunks(4).enumerate() {
                let mut b = [0u8; 4];
                b[..w.len()].copy_from_slice(w);
                let word = u32::from_le_bytes(b);
                let ascii: String = w
                    .iter()
                    .map(|c| if (32..127).contains(c) { *c as char } else { '.' })
                    .collect();
                println!("   &{:04X}: {word:08X}  {ascii}", n * 4);
            }
        }
        for r in &a.relocs {
            let by = match r.by {
                RelocBy::Area(n) => format!("area {n}"),
                RelocBy::Symbol(n) => format!("symbol {n}"),
            };
            let kind = match r.field {
                FieldType::Byte => "byte",
                FieldType::HalfWord => "halfword",
                FieldType::Word => "word",
                FieldType::Instruction => "instruction",
            };
            let mut how = vec![kind.to_string(), format!("by {by}")];
            if r.pc_relative {
                how.push("pc-relative".into());
            }
            if r.based {
                how.push("based".into());
            }
            if r.field == FieldType::Instruction && r.max_instructions != 0 {
                how.push(format!("{} instruction(s)", r.max_instructions));
            }
            println!("   reloc &{:04X}: {}", r.offset, how.join(", "));
        }
    }

    if !obj.symbols.is_empty() {
        println!();
        println!("** Symbol table");
        for s in &obj.symbols {
            let where_ = match &s.area {
                Some(a) => format!("{a} + &{:X}", s.value),
                None => format!("&{:X}", s.value),
            };
            println!("   {:<24} {where_:<24} {}", s.name, describe_symbol(s.attributes));
        }
    }
}
