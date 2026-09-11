//! `rosasm` — assemble ObjAsm source to an AOF object.
//!
//!     rosasm <source> -o <object> [-I dir]... [-PD "Sym SETA 1"]...
//!                      [--map <file>] [--keep-temps]
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
use rosasm::legalize::{self, AdrTarget, Legalized};
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
///
/// A name carrying a host separator is read as a host path and nothing else:
/// `GET ../VersionASM` and `GET ../../kernel/k_atomic.s` mean what they say,
/// and turning their dots into separators would make nonsense of them. The
/// type still moves, because `kernel/k_atomic.s` is `kernel.s.k_atomic`
/// however it is spelt.
fn relative_forms(name: &str) -> Vec<String> {
    let name = name.trim();
    if name.contains('/') || name.contains('\\') {
        let n = name.replace('\\', "/");
        let mut out = vec![n.clone()];
        if let Some((head, tail)) = n.rsplit_once('.') {
            if !tail.contains('/') {
                if let Some((dir, leaf)) = head.rsplit_once('/') {
                    out.push(format!("{dir}/{tail}/{leaf}"));
                }
            }
        }
        return out.into_iter().map(|f| parents(&f)).collect();
    }
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
    out.into_iter().map(|f| parents(&f)).collect()
}

/// `^` is RISC OS for the directory above, and the sources reach out of a
/// component with it: `GET ^.^.s.HeapMan` from `Kernel/Dev/HeapTest` is
/// `Kernel/s/HeapMan`.
fn parents(path: &str) -> String {
    if !path.contains('^') {
        return path.to_string();
    }
    path.split('/')
        .map(|p| if p == "^" { ".." } else { p })
        .collect::<Vec<_>>()
        .join("/")
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
fn to_ual(
    lines: &[ExpandedLine],
    ex: &Expander,
    refused: &mut Vec<Unencodable>,
    allow: bool,
    fpa_to_vfp: bool,
) -> (String, Vec<usize>, Vec<AdrReloc>) {
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
    let mut adr_relocs: Vec<AdrReloc> = Vec::new();
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
        // The operands as the expander froze them, not as the line reads:
        // a variable an operand names may have been rewritten since.
        let raw: &str = &l.operands;
        let raw = if rosasm::lower::is_swi(op) && !raw.trim_start().starts_with('#') {
            format!("#{raw}")
        } else {
            raw.to_string()
        };
        // Everything ObjAsm understands and LLVM does not is resolved here:
        // expressions, register aliases and bar-quoted names.
        let operands = ex.encoder_operands(l, op, &raw);
        let operands = rosasm::lower::translate_numbers(&operands);
        // One label per line lets the encoded bytes be matched back to the
        // line that produced them, whatever the instruction expands to.
        s.push_str(&format!("__ros{i}:\n"));
        // A literal the expander could not fold into the instruction lives in
        // a pool, and the pool is in this same area, so the distance to it is
        // fixed however the area is placed.
        if let Some(target) = l.literal {
            match pool_load(op, &operands, l.addr, target) {
                Ok(text) => {
                    s.push_str(&format!("        {text}\n"));
                    index.push(i);
                    continue;
                }
                Err(why) => {
                    s.push_str(&placeholder(op, refused.len(), allow));
                    refused.push(Unencodable::of(l, why));
                    index.push(i);
                    continue;
                }
            }
        }
        // Legalization decides what the instruction can become: itself, an
        // expansion, or nothing the encoder will accept.
        // An `ADR` at an imported symbol carries a relocation of its own:
        // the encoder is handed arithmetic on `pc` and has nothing to record.
        let external = adr_external(op, &operands, ex);
        if let Some(name) = external.clone() {
            adr_relocs.push(AdrReloc {
                line: i,
                name,
                instructions: rosasm::lower::instruction_words(op) as u8,
            });
        }
        let ctx = legalize::Context {
            here: l.addr,
            target: adr_target(op, &operands, l, ex),
            relocated: external.is_some(),
            fpa_to_vfp,
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
                // The space stays occupied either way, so later addresses do
                // not shift and the rest of the object stays readable.
                s.push_str(&placeholder(op, refused.len(), allow));
                refused.push(Unencodable::of(l, why));
            }
        }
        index.push(i);
    }
    (s, index, adr_relocs)
}

/// Something this assembler could not encode.
///
/// A zero word is `ANDEQ r0, r0, r0`: it executes, does nothing, and says
/// nothing, so an object holding one quietly does the wrong thing where an
/// instruction should have been. Every one is collected and the run fails
/// unless the caller has asked for otherwise.
struct Unencodable {
    /// Where the source said it, as `file:line`.
    where_: String,
    /// The line as written.
    what: String,
    why: String,
}

impl Unencodable {
    fn of(l: &ExpandedLine, why: String) -> Self {
        Self {
            where_: format!("{}:{}", l.origin.file, l.origin.line),
            what: l.text.trim().to_string(),
            why,
        }
    }
}

/// An `ADR` whose target only the linker knows.
struct AdrReloc {
    /// Index into the expanded lines, which is how its bytes are found again.
    line: usize,
    name: String,
    /// How many instructions the linker may rewrite: `ADRL` is two.
    instructions: u8,
}

/// Say what the encoder said, against the source rather than the lowering.
///
/// clang names a line in a temporary file, which is no use to anyone reading
/// it: the daily question is which line of which `.s` it came from. Every
/// instruction in the lowered text carries a `__ros<i>` label naming the
/// expanded line it came from, so walking back from the diagnostic to the
/// nearest label answers it.
fn report_encoder(stderr: &str, ual: &str, lines: &[ExpandedLine]) {
    let lowered: Vec<&str> = ual.lines().collect();
    for line in stderr.lines() {
        // `<path>:<line>:<col>: <severity>: <message>`. The path has a colon
        // of its own on this host, so the line is read from the right.
        let Some((head, severity, message)) = ["error", "warning", "note"]
            .iter()
            .find_map(|s| {
                let mark = format!(": {s}: ");
                line.find(&mark)
                    .map(|i| (&line[..i], *s, line[i + mark.len()..].trim()))
            })
        else {
            continue;
        };
        let mut fields = head.rsplit(':');
        let (Some(_col), Some(at)) = (fields.next(), fields.next()) else {
            continue;
        };
        let Ok(at) = at.trim().parse::<usize>() else { continue };
        match origin_of(&lowered, at, lines) {
            Some((where_, text)) => {
                eprintln!("rosasm: {where_}: {severity}: {message}");
                eprintln!("        {text}");
            }
            None => eprintln!("rosasm: {severity}: {message}"),
        }
    }
}

/// The source line an instruction in the lowered text came from.
fn origin_of<'a>(
    lowered: &[&str],
    at: usize,
    lines: &'a [ExpandedLine],
) -> Option<(String, &'a str)> {
    let mut i = at.min(lowered.len()).checked_sub(1)?;
    loop {
        if let Some(n) = lowered[i].strip_prefix("__ros").and_then(|t| t.strip_suffix(':')) {
            let l = lines.get(n.parse::<usize>().ok()?)?;
            return Some((format!("{}:{}", l.origin.file, l.origin.line), l.text.trim()));
        }
        i = i.checked_sub(1)?;
    }
}

/// The space an instruction was given, filled with something that says so.
///
/// As many words as the location counter reserved: two for `ADRL`, two for
/// an FPA compare, one for everything else. Emitting a single word instead
/// moves every label after it.
///
/// `UDF #n` traps where a zero word would have run on, and `n` says which
/// of the listed instructions it stands for.
fn placeholder(mnemonic: &str, n: usize, allow: bool) -> String {
    let words = rosasm::lower::instruction_words(mnemonic);
    if allow {
        format!("        UDF #{n}\n").repeat(words)
    } else {
        "        .inst 0x00000000\n".repeat(words)
    }
}

/// An `LDR Rd,=value` rendered as a load from the literal pool.
///
/// The offset is twelve bits with a sign, so a pool more than 4KB away is out
/// of reach -- which is the whole reason `LTORG` exists, and the error says so.
fn pool_load(mnemonic: &str, operands: &str, here: u32, target: u32) -> Result<String, String> {
    let rd = operands
        .split_once('=')
        .map(|(head, _)| head.trim().trim_end_matches(',').trim())
        .unwrap_or("r0");
    let delta = target as i64 - here as i64;
    // `pc` reads eight ahead, and the encoder works that out from `.` itself.
    if !(-4087..=4103).contains(&delta) {
        return Err(format!(
            "the literal pool is {delta} bytes away; an LDR reaches 4KB, so this \
             needs an LTORG nearer the instruction"
        ));
    }
    let m = rosasm::lower::normalise_mnemonic(mnemonic).unwrap_or_else(|| mnemonic.to_string());
    Ok(if delta < 0 {
        format!("{m} {rd}, .-{}", -delta)
    } else {
        format!("{m} {rd}, .+{delta}")
    })
}

/// What an `ADR`/`ADRL` is aiming at, when we can supply it.
///
/// The manual gives three kinds of expression -- register-relative,
/// program-relative and numeric -- and each becomes a different instruction,
/// so working out which it is happens here rather than in the encoding.
///
/// By the time the operands reach here, `encoder_operands` has already turned
/// a label or a local label in this same area into an offset from `.`, which
/// for this instruction is its own address. That is what marks the expression
/// as program-relative, and it also has to be read that way rather than
/// evaluated, because `.` in the symbol table holds wherever the location
/// counter finished, not where this line is.
///
/// A target in another area has no fixed distance from here, so it is refused
/// rather than expanded into something that would only be right by accident.
/// The imported symbol an `ADR` reaches for, if that is what it names.
///
/// `ADRL ip, cpuclock_Activate` in BCMSupport's device veneers: the symbol
/// is `IMPORT`ed, so its address is not known here and cannot be. ObjAsm
/// assembles the address as zero and leaves a relocation on the pair of
/// instructions for the linker to finish, which is what this makes possible.
fn adr_external(op: &str, operands: &str, ex: &Expander) -> Option<String> {
    if !rosasm::lower::is_adr(op) && !rosasm::lower::is_adrl(op) {
        return None;
    }
    let target = operands.split_once(',')?.1.trim();
    let names = expand::identifiers(target);
    let mut wanted = names.iter().filter(|n| ex.imports().contains(n));
    let name = wanted.next()?.clone();
    // One is a relocation; two in one expression is a distance the linker has
    // no way to compute.
    wanted.next().is_none().then_some(name)
}

fn adr_target(op: &str, operands: &str, l: &ExpandedLine, ex: &Expander) -> Option<AdrTarget> {
    if !rosasm::lower::is_adr(op) && !rosasm::lower::is_adrl(op) {
        return None;
    }
    let target = operands.split_once(',')?.1.trim();

    // An imported symbol has no address here. ObjAsm assembles the expression
    // with it standing at zero and relocates; the instructions come out the
    // same either way, and the relocation supplies the rest.
    if let Some(name) = adr_external(op, operands, ex) {
        let text = target.replace(&name, "0");
        return match rosasm::expr::eval(&text, ex.symbols()) {
            Ok(rosasm::symtab::Value::Arith(n)) => Some(AdrTarget::Program(n)),
            _ => None,
        };
    }

    // Program-relative: `.`, or an expression built on it such as the
    // `dtanid + (16 * 3)` the sources write.
    if target.starts_with('.') {
        let rest = &target[1..];
        if rest.is_empty() || rest.starts_with(['+', '-']) {
            let text = format!("{}{rest}", l.addr);
            return match rosasm::expr::eval(&text, ex.symbols()) {
                Ok(rosasm::symtab::Value::Arith(n)) => Some(AdrTarget::Program(n)),
                _ => None,
            };
        }
        return None;
    }

    let names = expand::identifiers(target);
    for name in &names {
        if let Some((area, _)) = ex.label_defs().get(name) {
            if *area != l.area_index {
                eprintln!(
                    "rosasm: {}:{}: {op} reaches into another area, which has no fixed distance",
                    l.origin.file, l.origin.line
                );
                return None;
            }
        }
    }
    let value = match rosasm::expr::eval(target, ex.symbols()) {
        Ok(rosasm::symtab::Value::Arith(n)) => n,
        _ => return None,
    };
    // Register-relative: a `MAP expr,Rn` made this symbol an offset from Rn.
    // Two symbols with different bases in one expression is meaningless, so
    // that is left unresolved rather than guessed at.
    let bases: Vec<u32> = names
        .iter()
        .filter_map(|n| ex.field_bases().get(n).copied())
        .collect();
    if let Some(base) = bases.first() {
        if bases.iter().all(|b| b == base) {
            // A storage map may be based below its register, so the offset
            // reads as signed.
            return Some(AdrTarget::Register { base: *base, offset: value as i32 });
        }
        return None;
    }
    // Numeric: not relative to anything, so it is moved rather than added.
    Some(AdrTarget::Numeric(value))
}

/// Where one run of the encoder's output ended up in an area.
struct Segment {
    /// Byte range within the encoder's `.text`.
    text: std::ops::Range<usize>,
    /// Index of the area it was copied into, and the offset there.
    area: usize,
    dest: u32,
    /// The expanded line it came from.
    line: usize,
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
    let mut map: Option<PathBuf> = None;
    let mut warn_assertions = false;
    let mut allow_unencodable = false;
    let mut fpa_to_vfp = false;
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
            // The sources assert their own layout, so a failure is a defect
            // and stops the build. Investigating one needs the opposite.
            "--warn-assertions" => warn_assertions = true,
            // An object with instructions missing from it is not an object
            // a ROM can be built with, so saying so is the default. This
            // asks for one anyway, with every gap trapping at run time.
            "--allow-unencodable" => allow_unencodable = true,
            // Not for building this ROM, where FPEmulator reads the FPA word
            // back and interprets it. For asking what the sources would look
            // like against the floating point the hardware has.
            "--fpa-to-vfp" => fpa_to_vfp = true,
            "--map" => {
                i += 1;
                map = args.get(i).map(PathBuf::from);
            }
            s => source = Some(PathBuf::from(s)),
        }
        i += 1;
    }
    let (Some(source), Some(out)) = (source, out) else {
        eprintln!(
            "usage: rosasm <source> -o <object> [-I dir]... [-PD assignment]... \
             [--map file] [--warn-assertions] [--allow-unencodable]              [--fpa-to-vfp] [--keep-temps]"
        );
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
    ex.set_assert_warnings(warn_assertions);
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

    // Where every line ended up. Enough to find a layout disagreement
    // against ObjAsm's listing without needing a listing of our own, which is
    // how the Kernel's `ASSERT {PC}-SVCDespatcher = SWIDespatch_Size` gets
    // tracked down: the first address that differs is the line that did it.
    if let Some(path) = &map {
        let mut out = String::from("; index  address  area  file:line  source\n");
        for (i, l) in lines.iter().enumerate() {
            if l.listing_only {
                continue;
            }
            out.push_str(&format!(
                "{i:<6} {:08X} {:>3}  {}:{}  {}\n",
                l.addr,
                l.area_index,
                l.origin.file,
                l.origin.line,
                l.text.trim_end()
            ));
        }
        if let Err(e) = std::fs::write(path, out) {
            eprintln!("rosasm: {}: {e}", path.display());
        }
    }

    // Encode the instructions.
    // Everything the object cannot honestly contain, collected rather
    // than printed and forgotten.
    let mut refused: Vec<Unencodable> = Vec::new();
    let (ual, index, adr_relocs) =
        to_ual(&lines, &ex, &mut refused, allow_unencodable, fpa_to_vfp);
    let tmp = std::env::temp_dir().join(format!("rosasm-{}", std::process::id()));
    let asm_path = tmp.with_extension("s");
    let obj_path = tmp.with_extension("o");
    if let Err(e) = std::fs::write(&asm_path, &ual) {
        eprintln!("rosasm: {e}");
        std::process::exit(1);
    }
    let run = Command::new(CLANG)
        .args(TARGET)
        .arg("-c")
        .arg(&asm_path)
        .arg("-o")
        .arg(&obj_path)
        .output();
    match run {
        Ok(out) if out.status.success() => {
            // Warnings still say something worth hearing, and they name the
            // lowered file too.
            report_encoder(&String::from_utf8_lossy(&out.stderr), &ual, &lines);
        }
        Ok(out) => {
            report_encoder(&String::from_utf8_lossy(&out.stderr), &ual, &lines);
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
    // Where each area changes between code and data, for the mapping symbols
    // ObjAsm emits: `$a` where ARM instructions start, `$d` where data does.
    // A disassembler cannot tell them apart without these, and the linker
    // uses them to decide what it may not reorder.
    let mut mapping: Vec<(usize, u32, char)> = Vec::new();
    let note = |area: usize, at: u32, kind: char, m: &mut Vec<(usize, u32, char)>| {
        if m.iter().rev().find(|(a, _, _)| *a == area).map(|(_, _, k)| *k) != Some(kind) {
            m.push((area, at, kind));
        }
    };
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
                note(l.area_index, area.data.len() as u32, 'd', &mut mapping);
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
                note(l.area_index, area.data.len() as u32, 'a', &mut mapping);
                segments.push(Segment {
                    text: from..to,
                    area: l.area_index,
                    dest: area.data.len() as u32,
                    line: i,
                });
                area.data.extend_from_slice(&text[from..to]);
            }
        }
    }
    // Anything that moved the location counter without emitting bytes leaves a
    // hole -- `SPACE` most of all, which is how SDFS reserves its stack. An
    // initialised area has to carry every byte it declares, so the holes are
    // filled here and the declared size becomes the data itself.
    for (a, size) in areas.iter_mut().zip(ex.area_sizes()) {
        if a.attributes & area_attr::ZERO_INIT != 0 {
            continue;
        }
        if a.data.len() > *size as usize {
            eprintln!(
                "rosasm: area {} holds {} bytes but its location counter reached {size}",
                a.name,
                a.data.len()
            );
        }
        a.data.resize((*size as usize).max(a.data.len()), 0);
        a.reserved = 0;
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
    // Every area carries its own name as a local symbol, and the points where
    // it changes between code and data carry `$a` and `$d`. Local, because
    // they describe the object rather than offering anything to the linker to
    // resolve; ObjAsm emits both and a disassembler expects them.
    for a in &areas {
        symbols.push(aof::Symbol {
            name: a.name.clone(),
            attributes: sym_attr::DEFINED,
            value: 0,
            area: Some(a.name.clone()),
        });
    }
    for (area, at, kind) in &mapping {
        let Some(a) = areas.get(*area) else { continue };
        symbols.push(aof::Symbol {
            name: format!("${kind}"),
            // `$d` says the bytes after it are a datum, which is exactly what
            // the <code datum> attribute records.
            attributes: sym_attr::DEFINED
                | if *kind == 'd' { sym_attr::CODE_DATUM } else { 0 },
            value: *at,
            area: Some(a.name.clone()),
        });
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
        // Only a symbol the encoder could not resolve becomes a relocation.
        // A `b .+20` is resolved where it stands, and LLVM still records a
        // fixup against a temporary of its own -- `.L0` -- which is already
        // accounted for in the bytes. Re-applying it corrupts the branch, and
        // puts a symbol in the object that names nothing.
        let defined = elf
            .symbols
            .get(r.sym as usize)
            .is_some_and(|s| s.is_defined());
        if defined || name.starts_with("__ros") || name.starts_with(".L") {
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
            refused.push(Unencodable {
                where_: lines.get(seg.line).map_or_else(
                    || format!("&{here:X}"),
                    |l| format!("{}:{}", l.origin.file, l.origin.line),
                ),
                what: lines.get(seg.line).map(|l| l.text.trim().to_string()).unwrap_or_default(),
                why: format!("relocation type {} against {name} is not handled", r.kind),
            });
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
                refused.push(Unencodable {
                    where_: lines.get(seg.line).map_or_else(
                        || format!("&{here:X}"),
                        |l| format!("{}:{}", l.origin.file, l.origin.line),
                    ),
                    what: lines.get(seg.line).map(|l| l.text.trim().to_string()).unwrap_or_default(),
                    why: format!("{name} is out of reach from here"),
                });
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

    // `ADR` at an imported symbol. The instructions are arithmetic on `pc`
    // with the symbol taken to stand at zero, so the encoder had nothing to
    // record and the relocation is added here, against the first of them.
    for a in &adr_relocs {
        let Some(seg) = segments.iter().find(|s| s.line == a.line) else {
            continue;
        };
        let idx = symbol_index(&mut symbols, &a.name);
        let Some(area) = areas.get_mut(seg.area) else { continue };
        area.relocs.push(aof::Reloc {
            offset: seg.dest,
            by: aof::RelocBy::Symbol(idx),
            field: aof::FieldType::Instruction,
            pc_relative: true,
            based: false,
            max_instructions: a.instructions,
        });
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

    // An object with instructions missing from it is not one a ROM can be
    // built with. Saying so, and writing nothing, is the default; the
    // alternative is asked for by name and traps at run time instead.
    if !refused.is_empty() {
        for (n, r) in refused.iter().enumerate() {
            let index = if allow_unencodable {
                format!(" [UDF #{n}]")
            } else {
                String::new()
            };
            eprintln!("rosasm: {}: {}{index}", r.where_, r.why);
            eprintln!("        {}", r.what);
        }
        let n = refused.len();
        let s = if n == 1 { "" } else { "s" };
        if allow_unencodable {
            eprintln!("rosasm: {n} instruction{s} will trap if reached");
        } else {
            eprintln!("rosasm: {n} instruction{s} could not be encoded");
            eprintln!("        no object written");
            eprintln!("        --allow-unencodable writes one anyway, trapping at each");
            std::process::exit(1);
        }
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
