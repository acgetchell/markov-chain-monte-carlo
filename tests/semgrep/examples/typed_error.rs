use markov_chain_monte_carlo::McmcError;

fn typed_example_error() -> Result<(), McmcError> {
    Ok(())
}

fn borrowed_typed_example_error(error: &McmcError) {
    let _ = error.to_string();
}
