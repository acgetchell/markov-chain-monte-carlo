fn erased_benchmark_error() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

fn borrowed_benchmark_error(error: &dyn std::error::Error) {
    let _ = error.to_string();
}

fn anyhow_benchmark_error(error: anyhow::Error) {
    let _ = error.to_string();
}
