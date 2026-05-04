/// ```
/// # use markov_chain_monte_carlo::McmcError;
/// # Ok::<(), McmcError>(())
/// ```
pub fn doctest_with_typed_error() {}

/**
 * ```
 * # use markov_chain_monte_carlo::McmcError;
 * # fn borrowed(error: &McmcError) {
 * #     let _ = error.to_string();
 * # }
 * ```
 */
pub fn block_doctest_with_typed_error() {}
