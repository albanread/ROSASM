//! ObjAsm line lexer.
//!
//! ObjAsm is column-sensitive in exactly one way: a non-blank character in
//! column 1 starts a label, and there is no trailing colon. Everything else is
//! whitespace-separated fields:
//!
//! ```text
//! [label] <ws> [opcode] <ws> [operands] [; comment]
//! ```
//!
//! Any field may be absent. The only real subtlety is finding where the
//! operands stop and the comment starts, because `;` is ordinary text inside a
//! string literal or a character constant — `DCB "a;b"` and `CMP r0,#';'` are
//! both common in the corpus.

/// Half-open range of char indices into the raw line.
pub type Span = (usize, usize);

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// Nothing but whitespace.
    Blank,
    /// Whole line is a comment (`;` in column 1, or only whitespace before it).
    Comment,
    /// A real statement. Any of label/opcode/operands may still be absent —
    /// a bare label on its own line is a statement with no opcode.
    Statement,
}

#[derive(Debug, Clone)]
pub struct Line {
    pub num: usize,
    pub raw: String,
    pub kind: Kind,
    pub label: Option<Span>,
    pub opcode: Option<Span>,
    pub operands: Option<Span>,
    pub comment: Option<Span>,
}

impl Line {
    pub fn text(&self, s: Span) -> &str {
        // Spans are char indices; the corpus is Latin-1 so chars may be
        // multi-byte in the decoded String. Slice by chars, not bytes.
        let start = self
            .raw
            .char_indices()
            .nth(s.0)
            .map(|(i, _)| i)
            .unwrap_or(self.raw.len());
        let end = self
            .raw
            .char_indices()
            .nth(s.1)
            .map(|(i, _)| i)
            .unwrap_or(self.raw.len());
        &self.raw[start..end]
    }

    pub fn label_str(&self) -> Option<&str> {
        self.label.map(|s| self.text(s))
    }
    pub fn opcode_str(&self) -> Option<&str> {
        self.opcode.map(|s| self.text(s))
    }
    pub fn operands_str(&self) -> Option<&str> {
        self.operands.map(|s| self.text(s))
    }
}

/// Find the char index where a comment starts, or None.
///
/// Skips over double-quoted strings (ObjAsm doubles an embedded quote: `""`)
/// and over character constants of the exact form `'x'`. A lone apostrophe —
/// common in English comments and in `DCB "don't"` — is treated as ordinary
/// text rather than opening a quote that swallows the rest of the line.
fn find_comment(chars: &[char]) -> Option<usize> {
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '"' => {
                i += 1;
                while i < chars.len() {
                    if chars[i] == '"' {
                        // `""` inside a string is an escaped quote.
                        if chars.get(i + 1) == Some(&'"') {
                            i += 2;
                            continue;
                        }
                        break;
                    }
                    i += 1;
                }
                i += 1;
            }
            '\'' if chars.get(i + 2) == Some(&'\'') => {
                // Character constant 'x' — including the awkward ';'.
                i += 3;
            }
            ';' => return Some(i),
            _ => i += 1,
        }
    }
    None
}

/// Lex one line.
pub fn lex_line(num: usize, raw: &str) -> Line {
    let chars: Vec<char> = raw.chars().collect();
    let mut line = Line {
        num,
        raw: raw.to_string(),
        kind: Kind::Blank,
        label: None,
        opcode: None,
        operands: None,
        comment: None,
    };

    // Split the comment off first; everything else parses within the remainder.
    let body_end = match find_comment(&chars) {
        Some(i) => {
            line.comment = Some((i, chars.len()));
            i
        }
        None => chars.len(),
    };

    let is_ws = |c: char| c == ' ' || c == '\t';

    // Anything left?
    let first_non_ws = (0..body_end).find(|&i| !is_ws(chars[i]));
    let Some(first) = first_non_ws else {
        line.kind = if line.comment.is_some() {
            Kind::Comment
        } else {
            Kind::Blank
        };
        return line;
    };

    line.kind = Kind::Statement;
    let mut pos = 0usize;

    // Column 1 non-blank => label field.
    if first == 0 {
        let end = (0..body_end).find(|&i| is_ws(chars[i])).unwrap_or(body_end);
        line.label = Some((0, end));
        pos = end;
    }

    // Opcode.
    let op_start = (pos..body_end).find(|&i| !is_ws(chars[i]));
    let Some(op_start) = op_start else { return line };
    let op_end = (op_start..body_end)
        .find(|&i| is_ws(chars[i]))
        .unwrap_or(body_end);
    line.opcode = Some((op_start, op_end));

    // Operands: everything to the comment, right-trimmed.
    let rest_start = (op_end..body_end).find(|&i| !is_ws(chars[i]));
    if let Some(s) = rest_start {
        let mut e = body_end;
        while e > s && is_ws(chars[e - 1]) {
            e -= 1;
        }
        line.operands = Some((s, e));
    }

    line
}

/// Join lines continued with a trailing `\`.
///
/// ObjAsm continues a statement when its code portion — the part before any
/// comment — ends in a backslash. Long `SETS` expressions built with `:CC:` do
/// this routinely. The joined statement keeps the line number it started on,
/// which is what diagnostics should report.
fn join_continuations(lines: &[String]) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = Vec::new();
    let mut pending: Option<(usize, String)> = None;

    for (i, raw) in lines.iter().enumerate() {
        let chars: Vec<char> = raw.chars().collect();
        let body_end = find_comment(&chars).unwrap_or(chars.len());
        let body: String = chars[..body_end].iter().collect();
        let trimmed = body.trim_end();
        let continued = trimmed.ends_with('\\');
        // Drop the backslash but keep the text before it.
        let piece = if continued {
            trimmed[..trimmed.len() - 1].to_string()
        } else {
            // Not continued: keep the whole raw line so the comment survives.
            raw.clone()
        };

        match pending.take() {
            Some((start, mut acc)) => {
                acc.push_str(&piece);
                if continued {
                    pending = Some((start, acc));
                } else {
                    out.push((start, acc));
                }
            }
            None => {
                if continued {
                    pending = Some((i + 1, piece));
                } else {
                    out.push((i + 1, piece));
                }
            }
        }
    }
    // A trailing `\` at end of file: keep what we have rather than losing it.
    if let Some((start, acc)) = pending {
        out.push((start, acc));
    }
    out
}

pub fn lex(lines: &[String]) -> Vec<Line> {
    join_continuations(lines)
        .into_iter()
        .map(|(num, text)| lex_line(num, &text))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l(s: &str) -> Line {
        lex_line(1, s)
    }

    #[test]
    fn blank_and_comment_lines() {
        assert_eq!(l("").kind, Kind::Blank);
        assert_eq!(l("    ").kind, Kind::Blank);
        assert_eq!(l("; just a comment").kind, Kind::Comment);
        assert_eq!(l("    ; indented comment").kind, Kind::Comment);
    }

    #[test]
    fn label_only_in_column_one() {
        let x = l("MyLabel");
        assert_eq!(x.label_str(), Some("MyLabel"));
        assert_eq!(x.opcode_str(), None);
    }

    #[test]
    fn opcode_must_be_indented() {
        let x = l("        MOV     r0, #1");
        assert_eq!(x.label_str(), None);
        assert_eq!(x.opcode_str(), Some("MOV"));
        assert_eq!(x.operands_str(), Some("r0, #1"));
    }

    #[test]
    fn label_opcode_operands() {
        let x = l("Loop    SUBS    r0, r0, #1      ; count down");
        assert_eq!(x.label_str(), Some("Loop"));
        assert_eq!(x.opcode_str(), Some("SUBS"));
        assert_eq!(x.operands_str(), Some("r0, r0, #1"));
        assert_eq!(x.text(x.comment.unwrap()), "; count down");
    }

    #[test]
    fn semicolon_inside_string_is_not_a_comment() {
        let x = l("        DCB     \"a;b\"   ; real comment");
        assert_eq!(x.operands_str(), Some("\"a;b\""));
        assert_eq!(x.text(x.comment.unwrap()), "; real comment");
    }

    #[test]
    fn semicolon_as_character_constant() {
        let x = l("        CMP     r0, #';'");
        assert_eq!(x.operands_str(), Some("r0, #';'"));
        assert!(x.comment.is_none());
    }

    #[test]
    fn apostrophe_in_text_does_not_swallow_comment() {
        let x = l("        DCB     \"don't\"  ; fine");
        assert_eq!(x.text(x.comment.unwrap()), "; fine");
    }

    #[test]
    fn doubled_quote_inside_string() {
        let x = l("        DCB     \"say \"\"hi\"\"\" ; c");
        assert_eq!(x.operands_str(), Some("\"say \"\"hi\"\"\""));
        assert_eq!(x.text(x.comment.unwrap()), "; c");
    }

    #[test]
    fn symbolic_opcodes_lex_as_opcodes() {
        assert_eq!(l("Sym     *       42").opcode_str(), Some("*"));
        assert_eq!(l("      [ :DEF: foo").opcode_str(), Some("["));
        assert_eq!(l("      ]").opcode_str(), Some("]"));
        assert_eq!(l("Field   #       4").opcode_str(), Some("#"));
        assert_eq!(l("      ^       0, r0").opcode_str(), Some("^"));
    }

    #[test]
    fn macro_parameter_label() {
        let x = l("$label  MOV     r0, $val");
        assert_eq!(x.label_str(), Some("$label"));
        assert_eq!(x.operands_str(), Some("r0, $val"));
    }

    #[test]
    fn tabs_separate_fields() {
        let x = l("Lbl\tMOV\tr0, #1");
        assert_eq!(x.label_str(), Some("Lbl"));
        assert_eq!(x.opcode_str(), Some("MOV"));
        assert_eq!(x.operands_str(), Some("r0, #1"));
    }
}

#[cfg(test)]
mod continuation_tests {
    use super::*;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn trailing_backslash_joins_the_next_line() {
        // As used by long SETS expressions built with :CC:.
        let src = v(&["Text SETS \"a\":CC: \\", "     \"b\""]);
        let out = lex(&src);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].num, 1, "joined statement keeps its first line number");
        assert!(out[0].operands_str().unwrap().contains("\"a\""));
        assert!(out[0].operands_str().unwrap().contains("\"b\""));
    }

    #[test]
    fn several_continuations_chain() {
        let src = v(&["A SETS \"1\" \\", "  \"2\" \\", "  \"3\"", "B SETS \"4\""]);
        let out = lex(&src);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].label_str(), Some("A"));
        assert_eq!(out[1].label_str(), Some("B"));
        assert_eq!(out[1].num, 4);
    }

    #[test]
    fn a_backslash_inside_a_comment_does_not_continue() {
        let src = v(&["    MOV r0, #1   ; a backslash \\", "    MOV r1, #2"]);
        let out = lex(&src);
        assert_eq!(out.len(), 2, "comment text must not join lines");
    }

    #[test]
    fn unterminated_continuation_at_eof_is_kept() {
        let src = v(&["A SETS \"x\" \\"]);
        assert_eq!(lex(&src).len(), 1);
    }
}
