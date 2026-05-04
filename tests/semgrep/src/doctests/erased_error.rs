/// ```
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn doctest_with_boxed_error() {}

/**
 * ```
 * # fn borrowed(error: &dyn std::error::Error) { let _ = error.to_string(); }
 * ```
 */
pub fn block_doctest_with_borrowed_error() {}

/*!
 * ```
 * # Ok::<(), anyhow::Error>(())
 * ```
 */
pub fn inner_block_doctest_with_anyhow_error() {}
