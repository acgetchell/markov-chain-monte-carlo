// ruleid: mcmc.rust.prefer-prelude-imports-in-examples
use markov_chain_monte_carlo::chain::Chain;

// ok: mcmc.rust.prefer-prelude-imports-in-examples
use markov_chain_monte_carlo::prelude::Sampler;

fn main() {
    let _ = std::any::type_name::<Chain<f64>>();
    let _ = std::any::type_name::<Sampler<'static, f64, (), (), ()>>();
}
