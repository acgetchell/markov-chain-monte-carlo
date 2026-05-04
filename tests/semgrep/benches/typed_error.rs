use markov_chain_monte_carlo::McmcError;

fn typed_benchmark_error() -> Result<(), McmcError> {
    Ok(())
}

fn borrowed_typed_benchmark_error(error: &McmcError) {
    let _ = error.to_string();
}
