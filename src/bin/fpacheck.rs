//! `fpacheck` — how much of the corpus's FPA code VFP can express.
//!
//!     fpacheck <path to RiscOS/Sources>
//!
//! Walks every source file, finds the FPA instructions, and runs each through
//! the same `fpa::convert` the assembler uses. Reports what translated, and
//! groups what did not by the reason it was refused.
//!
//! This reads the source text directly rather than expanding it, so a line
//! inside a false conditional is counted alongside one that assembles. That
//! overstates the total a little and understates nothing, which is the right
//! way round for a report about what cannot be translated.
//!
//! Two filters keep it from overstating wildly. Only files under an `s/`
//! directory are read, because the tree also holds C, BASIC and documentation
//! whose second word means nothing here. And a candidate must name an `f0`-`f7`
//! register, because plenty of ordinary symbols decode as FPA mnemonics if you
//! let them: `LOGGED` parses as `LOG` conditional on `GE` in double precision,
//! and it is a variable. The status transfers are the exception -- they take an
//! ARM register -- so those are matched by name.

use rosasm::fpa;
use rosasm::legalize::Legalized;
use rosasm::lex;
use rosasm::source::SourceFile;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Default)]
struct Tally {
    /// `FACC FN 0` names `f0`. The assembler learns these as it expands; this
    /// gathers them in a first pass instead, because a report that counted
    /// every `ADFD FACC,FACC,F1` as untranslatable would be wrong.
    fn_aliases: BTreeMap<String, u32>,
    translated: BTreeMap<String, usize>,
    refused: BTreeMap<String, usize>,
    /// Reason text to (count, an example mnemonic).
    reasons: BTreeMap<String, (usize, String)>,
    files: BTreeMap<String, usize>,
}

/// The reason without the mnemonic that prefixes it, so like refusals group.
fn reason_of(why: &str) -> String {
    match why.split_once(": ") {
        Some((_, rest)) => rest.trim().to_string(),
        None => why.to_string(),
    }
}

fn walk(dir: &Path, t: &mut Tally) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == ".git") {
                continue;
            }
            walk(&p, t);
        } else if is_assembler(&p) {
            if let Ok(sf) = SourceFile::load(&p) {
                scan(&p, &sf.lines, t);
            }
        }
    }
}

/// Gather every `FN` declaration in the tree, so an aliased register name is
/// recognised as the FPA register it stands for.
fn collect_fn(dir: &Path, t: &mut Tally) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if !p.file_name().is_some_and(|n| n == ".git") {
                collect_fn(&p, t);
            }
        } else if is_assembler(&p) {
            let Ok(sf) = SourceFile::load(&p) else { continue };
            for (n, raw) in sf.lines.iter().enumerate() {
                let l = lex::lex_line(n + 1, raw);
                if l.opcode_str().map(|o| o.eq_ignore_ascii_case("FN")) != Some(true) {
                    continue;
                }
                let (Some(name), Some(v)) = (l.label_str(), l.operands_str()) else { continue };
                if let Ok(v) = v.trim().parse::<u32>() {
                    if v <= 7 {
                        t.fn_aliases.insert(name.trim().to_string(), v);
                    }
                }
            }
        }
    }
}

/// Replace `FN`-declared names with the register they stand for.
fn resolve_aliases(operands: &str, aliases: &BTreeMap<String, u32>) -> String {
    let mut out = String::with_capacity(operands.len());
    let mut word = String::new();
    let flush = |w: &mut String, out: &mut String| {
        if !w.is_empty() {
            match aliases.get(w.as_str()) {
                Some(n) => out.push_str(&format!("f{n}")),
                None => out.push_str(w),
            }
            w.clear();
        }
    };
    for c in operands.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '$' {
            word.push(c);
        } else {
            flush(&mut word, &mut out);
            out.push(c);
        }
    }
    flush(&mut word, &mut out);
    out
}

/// Assembler lives in `s/` by the tree's own convention, and `hdr/` holds the
/// macro definitions it pulls in.
fn is_assembler(p: &Path) -> bool {
    p.parent()
        .and_then(Path::file_name)
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.eq_ignore_ascii_case("s") || n.eq_ignore_ascii_case("hdr"))
}

/// Does this operand list name an FPA register?
fn names_an_fpa_register(operands: &str) -> bool {
    let cs: Vec<char> = operands.chars().collect();
    for (i, c) in cs.iter().enumerate() {
        if *c != 'f' && *c != 'F' {
            continue;
        }
        // Whole word only: `fp` and `offset` are not `f0`.
        if i > 0 && (cs[i - 1].is_ascii_alphanumeric() || cs[i - 1] == '_') {
            continue;
        }
        match cs.get(i + 1) {
            Some(d) if d.is_ascii_digit() && *d <= '7' => {
                if !cs.get(i + 2).is_some_and(|n| n.is_ascii_alphanumeric() || *n == '_') {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn scan(path: &Path, lines: &[String], t: &mut Tally) {
    let mut here = 0usize;
    for (n, raw) in lines.iter().enumerate() {
        let l = lex::lex_line(n + 1, raw);
        let Some(op) = l.opcode_str() else { continue };
        if !fpa::is_fpa(op) {
            continue;
        }
        let operands = &resolve_aliases(l.operands_str().unwrap_or(""), &t.fn_aliases);
        let operands: &str = operands;
        let up = op.to_ascii_uppercase();
        // A macro body is a template, not a site: `ADFD $a, $b, $c` becomes a
        // real instruction only once it is invoked, and this does not expand.
        if operands.contains('$') {
            continue;
        }
        // The status transfers take an ARM register; everything else has to
        // name an FPA one, or it is an ordinary symbol that happens to decode.
        let status = matches!(up.as_str(), "RFS" | "WFS" | "RFC" | "WFC")
            || up.len() == 5 && matches!(&up[..3], "RFS" | "WFS" | "RFC" | "WFC");
        if !status && !names_an_fpa_register(operands) {
            continue;
        }
        match fpa::convert(&up, operands) {
            Some(Legalized::Unsupported(why)) => {
                *t.refused.entry(up.clone()).or_default() += 1;
                let r = reason_of(&why);
                let e = t.reasons.entry(r).or_insert((0, up));
                e.0 += 1;
            }
            Some(_) => *t.translated.entry(up).or_default() += 1,
            None => continue,
        }
        here += 1;
    }
    if here > 0 {
        *t.files.entry(path.display().to_string()).or_default() += here;
    }
}

fn main() {
    let Some(root) = std::env::args().nth(1) else {
        eprintln!("usage: fpacheck <path to RiscOS/Sources>");
        std::process::exit(2);
    };
    let mut t = Tally::default();
    collect_fn(Path::new(&root), &mut t);
    walk(Path::new(&root), &mut t);

    let translated: usize = t.translated.values().sum();
    let refused: usize = t.refused.values().sum();
    let total = translated + refused;
    if total == 0 {
        println!("no FPA instructions found under {root}");
        return;
    }
    let pct = |n: usize| 100.0 * n as f64 / total as f64;
    println!("FPA instructions      {total}");
    println!("  translated to VFP   {translated} ({:.1}%)", pct(translated));
    println!("  refused             {refused} ({:.1}%)", pct(refused));

    println!("\ntranslated:");
    let mut v: Vec<_> = t.translated.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (m, n) in v.iter().take(24) {
        println!("  {n:5}  {m}");
    }

    println!("\nrefused, by reason:");
    let mut r: Vec<_> = t.reasons.iter().collect();
    r.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    for (why, (n, example)) in r {
        println!("  {n:5}  {example}: {why}");
    }

    println!("\nfiles with FPA code:");
    let mut f: Vec<_> = t.files.iter().collect();
    f.sort_by(|a, b| b.1.cmp(a.1));
    for (p, n) in f.iter().take(20) {
        println!("  {n:5}  {p}");
    }
}
