fn erased_example_error() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

fn borrowed_example_error(error: &dyn std::error::Error) {
    let _ = error.to_string();
}

fn anyhow_example_error(error: anyhow::Error) {
    let _ = error.to_string();
}
