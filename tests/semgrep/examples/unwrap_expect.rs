#![allow(dead_code)]

fn example_fixture(result: Result<u32, &'static str>, value: Option<u32>) {
    // ruleid: mcmc.rust.no-unwrap-expect-in-benches-examples
    let _ = result.unwrap();

    // ruleid: mcmc.rust.no-unwrap-expect-in-benches-examples
    let _ = value.expect("examples should avoid expect");

    // ok: mcmc.rust.no-unwrap-expect-in-benches-examples
    let _ = result.unwrap_or(0);
}
