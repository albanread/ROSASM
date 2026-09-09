//! `rosasm` — an ObjAsm-compatible assembler front end for RISC OS 5.
//!
//! Division of labour: this crate owns everything above the mnemonic — the
//! macro language, conditional assembly, the three-valued symbol table,
//! directives and layout. Instruction encoding, relocation and object writing
//! are LLVM's, reached by lowering to UAL.
//!
//! Target is Raspberry Pi 4 (Cortex-A72, ARMv8-A in AArch32, NEON/VFPv4).
//! No 26-bit modes, no FPA.
//!
//! See `docs/ROSASM-DESIGN.md` for the design and its rationale.

pub mod aof;
pub mod elfread;
pub mod expand;
pub mod expr;
pub mod layout;
pub mod legalize;
pub mod lex;
pub mod listing;
pub mod lower;
pub mod reloc;
pub mod source;
pub mod symtab;
pub mod vocab;
