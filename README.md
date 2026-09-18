# rosasm

An ObjAsm-compatible assembler for RISC OS 5, written in Rust. It reads
the ObjAsm sources the RISC OS 5 components are written in and emits AOF
objects, or ELF32 ARM objects with `--elf`, for the Raspberry Pi 4's
Cortex-A72 running RISC OS in 32-bit ARM (AArch32). Experimental
software, unsupported.

## Build

Needs a Rust toolchain and clang, used as the encoder:

    cargo build --release

## Run

    target/release/rosasm <source> -o <object> [-I dir]...
    target/release/rosasm s/head -o head.o --elf

Licence: MIT — see LICENSE.

This repository is scheduled to be archived. Pull requests and issues are not accepted.
