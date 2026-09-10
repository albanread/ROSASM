fn main() {
    for m in ["MCR", "MCRNE", "MRC", "MCRR", "MRRC", "CDP"] {
        println!("{m:8} normalise={:?} is_fpa={} adr={} adrl={}",
            rosasm::lower::normalise_mnemonic(m),
            rosasm::fpa::is_fpa(m),
            rosasm::lower::is_adr(m),
            rosasm::lower::is_adrl(m));
    }
}
