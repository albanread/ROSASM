//! `rosasm` — assemble ObjAsm source to an AOF object.
//!
//!     rosasm <source> -o <object> [-I dir]... [-PD "Sym SETA 1"]...
//!
//! The pipeline: expand the macro language, lower each instruction to UAL,
//! hand that to LLVM's integrated assembler for encoding, then translate the
//! ELF it produces into AOF, which is what the RISC OS linker reads.
//!
//! LLVM is used only as an encoder. Everything above the mnemonic — the macro
//! language, conditional assembly, the symbol table, layout — is ours, and
//! everything below the object file is AOF, which LLVM knows nothing about.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use rosasm::aof::{self, area_attr, sym_attr};
use rosasm::elfread;
use rosasm::expand::{self, Expander, ExpandedLine, FileResolver};
use rosasm::legalize::{self, Legalized};
use rosasm::lex;
use rosasm::reloc;
use rosasm::source::SourceFile;

/// Where to find the encoder. clang drives LLVM's integrated assembler.
const CLANG: &str = r"C:\Program Files\LLVM\bin\clang.exe";
/// Pi 4: ARMv8-A in AArch32 with NEON.
const TARGET: &[&str] = &[
    "--target=arm-none-eabi",
    "-mcpu=cortex-a72",
    // The A72's floating point is VFPv4 with NEON; plain `neon` is VFPv3 and
    // rejects the fused multiply-adds the sources use.
    "-mfpu=neon-fp-armv8",
];

struct Dirs {
    dirs: Vec<PathBuf>,
    vars: HashMap<String, Vec<PathBuf>>,
}

/// RISC OS writes `dir.file`, but the sources also write `file.dir` — the DDE
/// resolves both, which is what lets one tree be read from a RISC OS or a
/// Unix-style host.
fn relative_forms(name: &str) -> Vec<String> {
    let name = name.trim();
    let mut out = vec![name.replace('.', "/")];
    if let Some((head, tail)) = name.rsplit_once('.') {
        let head = head.replace('.', "/");
        // `s.Foo` written the other way round.
        let swapped = format!("{tail}/{head}");
        if !out.contains(&swapped) {
            out.push(swapped);
        }
        // `clib.s.cl_data` written as `clib/cl_data.s`: the type directory
        // belongs immediately before the leaf, not at the front.
        let nested = match head.rsplit_once('/') {
            Some((dir, leaf)) => format!("{dir}/{tail}/{leaf}"),
            None => format!("{tail}/{head}"),
        };
        if !out.contains(&nested) {
            out.push(nested);
        }
    }
    out
}

impl FileResolver for Dirs {
    fn resolve(&self, name: &str) -> Option<(String, Vec<String>)> {
        let name = name.trim();
        let mut cands: Vec<PathBuf> = Vec::new();
        if let Some((var, rest)) = name.split_once(':') {
            if let Some(ds) = self.vars.get(&var.to_ascii_lowercase()) {
                for rel in relative_forms(rest) {
                    cands.extend(ds.iter().map(|d| d.join(&rel)));
                }
            }
        } else {
            for rel in relative_forms(name) {
                cands.extend(self.dirs.iter().map(|d| d.join(&rel)));
            }
        }
        for p in cands {
            if p.is_file() {
                if let Ok(sf) = SourceFile::load(&p) {
                    return Some((name.to_string(), sf.lines));
                }
            }
        }
        None
    }
}

/// Render the expanded lines as a UAL assembly file for the encoder.
///
/// Data directives are not re-emitted: their bytes were computed during
/// expansion, where `@`, `?label` and the ObjAsm operators are meaningful.
/// Only instructions go to LLVM.
fn to_ual(lines: &[ExpandedLine], ex: &Expander) -> (String, Vec<usize>) {
    // The directives have to agree with the command line, and they win where
    // they disagree: `.fpu neon` is VFPv3 and would refuse the A72's fused
    // multiply-adds however the driver was invoked.
    let mut s = String::from(
        "        .syntax unified\n\
         \x20       .arch armv8-a\n\
         \x20       .fpu neon-fp-armv8\n\
         \x20       .text\n",
    );
    let mut index = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if l.listing_only || !l.bytes.is_empty() {
            continue;
        }
        let lx = lex::lex_line(l.origin.line, &l.text);
        let Some(op) = lx.opcode_str() else { continue };
        if is_directive(op) {
            continue;
        }
        // `SWI OS_Write0` names the SWI, and the name is a symbol from a
        // header; UAL wants an immediate, so mark it as one to evaluate.
        let raw = lx.operands_str().unwrap_or("");
        let raw = if rosasm::lower::is_swi(op) && !raw.trim_start().starts_with('#') {
            format!("#{raw}")
        } else {
            raw.to_string()
        };
        // Everything ObjAsm understands and LLVM does not is resolved here:
        // expressions, register aliases and bar-quoted names.
        let operands = ex.encoder_operands(l, &raw);
        let operands = rosasm::lower::translate_numbers(&operands);
        // One label per line lets the encoded bytes be matched back to the
        // line that produced them, whatever the instruction expands to.
        s.push_str(&format!("__ros{i}:\n"));
        // Legalization decides what the instruction can become: itself, an
        // expansion, or nothing the encoder will accept.
        let ctx = legalize::Context {
            here: l.addr,
            target: adr_target(op, &operands, l, ex),
        };
        match legalize::legalize(op, &operands, &ctx) {
            Legalized::One(m, o) => s.push_str(&format!("        {m} {o}\n")),
            Legalized::Many(v) => {
                for (m, o) in v {
                    s.push_str(&format!("        {m} {o}\n"));
                }
            }
            Legalized::RawWord(w) => s.push_str(&format!("        .inst 0x{w:08X}\n")),
            Legalized::Unsupported(why) => {
                eprintln!("rosasm: {}:{}: {why}", l.origin.file, l.origin.line);
                // Keep the space occupied so later addresses do not shift.
                s.push_str("        .inst 0x00000000\n");
            }
        }
        index.push(i);
    }
    (s, index)
}

/// The address an `ADR`/`ADRL` is aiming at, when we can supply it.
///
/// These are pseudo-instructions the encoder cannot expand, because the label
/// is ours and its value is an offset within an AOF area. A target in another
/// area has no fixed distance from here, so it is refused rather than expanded
/// into something that would only be right by accident.
fn adr_target(op: &str, operands: &str, l: &ExpandedLine, ex: &Expander) -> Option<u32> {
    if !rosasm::lower::is_adr(op) && !rosasm::lower::is_adrl(op) {
        return None;
    }
    let target = operands.split(',').nth(1)?.trim();
    for name in expand::identifiers(target) {
        match ex.label_defs().get(&name) {
            Some((area, _)) if *area == l.area_index => {}
            Some(_) => {
                eprintln!(
                    "rosasm: {}:{}: {op} reaches into another area, which has no fixed distance",
                    l.origin.file, l.origin.line
                );
                return None;
            }
            None => {}
        }
    }
    match rosasm::expr::eval(target, ex.symbols()) {
        Ok(rosasm::symtab::Value::Arith(n)) => Some(n),
        _ => None,
    }
}

/// Where one run of the encoder's output ended up in an area.
struct Segment {
    /// Byte range within the encoder's `.text`.
    text: std::ops::Range<usize>,
    /// Index of the area it was copied into, and the offset there.
    area: usize,
    dest: u32,
}

fn word_at(b: &[u8], off: u32) -> u32 {
    let i = off as usize;
    if i + 4 > b.len() {
        0
    } else {
        u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
    }
}

fn set_word_at(b: &mut [u8], off: u32, w: u32) {
    let i = off as usize;
    if i + 4 <= b.len() {
        b[i..i + 4].copy_from_slice(&w.to_le_bytes());
    }
}

fn field_type(width: u8) -> Option<aof::FieldType> {
    match width {
        1 => Some(aof::FieldType::Byte),
        2 => Some(aof::FieldType::HalfWord),
        4 => Some(aof::FieldType::Word),
        _ => None,
    }
}

/// Index of `name` in the symbol table, adding it as an external reference if
/// the source never mentioned it -- which happens whenever a branch names a
/// symbol the source neither defines nor imports.
fn symbol_index(symbols: &mut Vec<aof::Symbol>, name: &str) -> u32 {
    if let Some(i) = symbols.iter().position(|s| s.name == name) {
        return i as u32;
    }
    symbols.push(external(name));
    (symbols.len() - 1) as u32
}

/// An undefined global: the linker resolves it against another object.
fn external(name: &str) -> aof::Symbol {
    aof::Symbol {
        name: name.to_string(),
        attributes: sym_attr::GLOBAL,
        value: 0,
        area: None,
    }
}

fn is_directive(op: &str) -> bool {
    rosasm::vocab::is_directive(op) || rosasm::vocab::is_symbolic(op)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut source: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut pds: Vec<String> = Vec::new();
    let mut keep = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                out = args.get(i).map(PathBuf::from);
            }
            "-I" | "-i" => {
                i += 1;
                if let Some(d) = args.get(i) {
                    dirs.push(PathBuf::from(d));
                }
            }
            "-PD" | "-pd" => {
                i += 1;
                if let Some(d) = args.get(i) {
                    pds.push(d.clone());
                }
            }
            "--keep-temps" => keep = true,
            s => source = Some(PathBuf::from(s)),
        }
        i += 1;
    }
    let (Some(source), Some(out)) = (source, out) else {
        eprintln!("usage: rosasm <source> -o <object> [-I dir]... [-PD assignment]...");
        std::process::exit(2);
    };

    let sf = match SourceFile::load(&source) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rosasm: {}: {e}", source.display());
            std::process::exit(1);
        }
    };

    // `GET` resolves against the assembler's working directory, which the
    // makefiles set to the component root: RISC_OSLib is built with
    // `objasm -from clib.s.cl_stub`, so that file's `GET h_regs.s` finds
    // `RISC_OSLib/s/h_regs` -- two levels up, not one. Both depths are
    // searched, nearest first.
    let own = source.parent().map(Path::to_path_buf).unwrap_or_default();
    let comp = own.parent().map(Path::to_path_buf).unwrap_or_default();
    let above = comp.parent().map(Path::to_path_buf);
    let mut search = dirs.clone();
    search.push(comp.clone());
    search.extend(above.clone());
    // The component's own hdr first, as the build's Hdr$Path has it.
    let mut hdr = vec![comp.join("hdr")];
    hdr.extend(above.iter().map(|d| d.join("hdr")));
    hdr.extend(dirs.iter().cloned());
    let resolver = Dirs {
        dirs: search,
        vars: HashMap::from([("hdr".to_string(), hdr)]),
    };

    let mut ex = Expander::new(&resolver);
    ex.set_target_builtins();
    for pd in &pds {
        if let Err(e) = ex.predefine(pd) {
            eprintln!("rosasm: bad -PD {pd:?}: {e}");
            std::process::exit(1);
        }
    }
    let name = source.file_name().unwrap_or_default().to_string_lossy().to_string();
    let lines = match ex.run(&name, sf.lines) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // Encode the instructions.
    let (ual, index) = to_ual(&lines, &ex);
    let tmp = std::env::temp_dir().join(format!("rosasm-{}", std::process::id()));
    let asm_path = tmp.with_extension("s");
    let obj_path = tmp.with_extension("o");
    if let Err(e) = std::fs::write(&asm_path, &ual) {
        eprintln!("rosasm: {e}");
        std::process::exit(1);
    }
    let status = Command::new(CLANG)
        .args(TARGET)
        .arg("-c")
        .arg(&asm_path)
        .arg("-o")
        .arg(&obj_path)
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(_) => {
            eprintln!("rosasm: the encoder rejected the lowered assembly");
            eprintln!("        kept at {}", asm_path.display());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("rosasm: cannot run the encoder at {CLANG}: {e}");
            std::process::exit(1);
        }
    }

    let elf_bytes = std::fs::read(&obj_path).unwrap_or_default();
    let elf = match elfread::parse(&elf_bytes) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("rosasm: cannot read the encoder's output: {e}");
            std::process::exit(1);
        }
    };

    // Map each encoded instruction back to the line that produced it, so the
    // area is assembled in source order with the data already computed.
    let text = elf.section_named(".text").map(|s| s.data.clone()).unwrap_or_default();
    let mut at: HashMap<usize, u32> = HashMap::new();
    for s in &elf.symbols {
        if let Some(n) = s.name.strip_prefix("__ros") {
            if let Ok(n) = n.parse::<usize>() {
                at.insert(n, s.value);
            }
        }
    }

    // One AOF area per source AREA, in declaration order.
    let mut areas: Vec<aof::Area> = ex
        .areas()
        .iter()
        .zip(ex.area_sizes())
        .map(|((name, attrs), size)| {
            let mut a = aof::Area::new(name.clone(), aof::from_objasm_area(attrs));
            a.alignment = attrs.align as u8;
            // A NOINIT area emits nothing, so its size has to be declared.
            a.reserved = *size;
            a
        })
        .collect();
    if areas.is_empty() {
        // A source with no AREA at all still has to go somewhere.
        areas.push(aof::Area::new(
            "C$$code",
            area_attr::CODE | area_attr::READ_ONLY | area_attr::APCS_32,
        ));
    }

    // Copy each line's bytes into its area, remembering where the encoder's
    // output landed so its fixups can be found again.
    let mut segments: Vec<Segment> = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if l.listing_only {
            continue;
        }
        let Some(area) = areas.get_mut(l.area_index) else { continue };
        // A zero-initialised area carries no bytes in the file; its size comes
        // from the header alone, so anything emitted into one is dropped here.
        let zero_init = area.attributes & area_attr::ZERO_INIT != 0;
        if !l.bytes.is_empty() {
            if !zero_init {
                area.data.extend_from_slice(&l.bytes);
            }
            continue;
        }
        // How many bytes this line produced is the distance to the next
        // labelled line -- which is how an ADRL expansion contributes its
        // eight bytes without the copy needing to know about it.
        if let Some(off) = at.get(&i) {
            let from = *off as usize;
            let to = at
                .values()
                .map(|v| *v as usize)
                .filter(|v| *v > from)
                .min()
                .unwrap_or(text.len());
            if !zero_init && to <= text.len() && from < to {
                segments.push(Segment {
                    text: from..to,
                    area: l.area_index,
                    dest: area.data.len() as u32,
                });
                area.data.extend_from_slice(&text[from..to]);
            }
        }
    }
    // The spec requires each area's length to be a multiple of four.
    for a in &mut areas {
        while a.data.len() % 4 != 0 {
            a.data.push(0);
        }
    }
    let _ = index;

    // Exported labels become defining occurrences at the address the second
    // pass gave them; anything exported without a definition here, and every
    // import, becomes an external reference for the linker to satisfy.
    let defs = ex.label_defs();
    let mut symbols: Vec<aof::Symbol> = Vec::new();
    for name in ex.exports() {
        match defs.get(name) {
            Some((ai, off)) => symbols.push(aof::Symbol {
                name: name.clone(),
                attributes: sym_attr::DEFINED | sym_attr::GLOBAL,
                value: *off,
                area: areas.get(*ai).map(|a| a.name.clone()),
            }),
            None => {
                eprintln!("rosasm: {name} is exported but not defined here");
                symbols.push(external(name));
            }
        }
    }
    for name in ex.imports() {
        if !symbols.iter().any(|s| s.name == *name) {
            symbols.push(external(name));
        }
    }

    // Relocations. The encoder reports every field it could not fix; each is
    // either an addend we can work out here or a directive for the linker.
    let text_section = elf
        .sections
        .iter()
        .position(|s| s.name == ".text")
        .unwrap_or(0) as u32;
    for r in &elf.rels {
        if r.section != text_section {
            continue;
        }
        let Some(seg) = segments
            .iter()
            .find(|s| s.text.contains(&(r.offset as usize)))
        else {
            continue;
        };
        let name = elf
            .symbols
            .get(r.sym as usize)
            .map(|s| s.name.clone())
            .unwrap_or_default();
        if name.starts_with("__ros") {
            continue;
        }
        let here = seg.dest + (r.offset - seg.text.start as u32);
        let insn = word_at(&areas[seg.area].data, here);
        // Two field shapes reach us: a branch's 24-bit word offset and a data
        // transfer's 12-bit byte offset. Both are measured from `pc`.
        let (addend, encode): (i32, fn(u32, i32) -> Option<u32>) = if reloc::is_branch(r.kind) {
            (reloc::branch_addend(insn), reloc::set_branch_addend)
        } else if reloc::is_ldr_literal(r.kind) {
            (reloc::ldr_addend(insn), reloc::set_ldr_addend)
        } else {
            eprintln!(
                "rosasm: {name}: unhandled relocation type {} at &{here:X}",
                r.kind
            );
            continue;
        };
        // Three cases: a target in this same area needs no directive at all,
        // one in another of our areas is relocated by that area's base, and
        // anything else is relocated by the symbol's value.
        let (new_addend, by) = match defs.get(&name) {
            Some((ai, off)) if *ai == seg.area => (reloc::local_pc_addend(here, *off), None),
            Some((ai, off)) => (
                reloc::pc_relative_addend(addend, here, *off),
                Some(aof::RelocBy::Area(*ai as u32)),
            ),
            None => {
                let idx = symbol_index(&mut symbols, &name);
                (
                    reloc::pc_relative_addend(addend, here, 0),
                    Some(aof::RelocBy::Symbol(idx)),
                )
            }
        };
        match encode(insn, new_addend) {
            Some(w) => set_word_at(&mut areas[seg.area].data, here, w),
            None => {
                eprintln!("rosasm: {name} at &{here:X} is out of reach from here");
                continue;
            }
        }
        if let Some(by) = by {
            areas[seg.area].relocs.push(aof::Reloc {
                offset: here,
                by,
                field: aof::FieldType::Instruction,
                pc_relative: true,
                based: false,
                max_instructions: 1,
            });
        }
    }

    // Data fields the expander could not finish.
    for f in ex.data_fixups() {
        let Some(field) = field_type(f.width) else {
            eprintln!(
                "rosasm: cannot relocate a {}-byte field ({})",
                f.width, f.expr
            );
            continue;
        };
        let by = match &f.kind {
            expand::FixupKind::AreaBase(a) => aof::RelocBy::Area(*a as u32),
            expand::FixupKind::External(name) => {
                aof::RelocBy::Symbol(symbol_index(&mut symbols, name))
            }
        };
        let Some(area) = areas.get_mut(f.area) else { continue };
        area.relocs.push(aof::Reloc {
            offset: f.offset,
            by,
            field,
            // Plain additive: the value is an address, not a distance.
            pc_relative: false,
            based: false,
            max_instructions: 0,
        });
    }

    let obj = aof::Object {
        areas,
        symbols,
        entry: None,
        identification: format!("rosasm {}", env!("CARGO_PKG_VERSION")),
    };
    if let Err(e) = std::fs::write(&out, obj.write()) {
        eprintln!("rosasm: {}: {e}", out.display());
        std::process::exit(1);
    }
    if !keep {
        let _ = std::fs::remove_file(&asm_path);
        let _ = std::fs::remove_file(&obj_path);
    }
    let bytes: usize = obj.areas.iter().map(|a| a.data.len()).sum();
    let n = obj.areas.len();
    println!(
        "{}: {bytes} bytes in {n} area{}, {} symbol{}",
        out.display(),
        if n == 1 { "" } else { "s" },
        obj.symbols.len(),
        if obj.symbols.len() == 1 { "" } else { "s" }
    );
}
