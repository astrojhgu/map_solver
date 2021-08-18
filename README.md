# map_solver

## Install `rust` environment (I promise it is EASY)
1. Follow [rustup](https://rustup.rs) to install `rustup`
(alternatively, you can install rustup from the package manager of your os)
2. select the default `rust` compiler toolchain
```
rustup toolchain stable
```

## Get the source code
```
git checkout https://github.com/astrojhgu/map_solver.git
```

## Compile the code
```
cd map_solver
cargo build --release
```

## Run the code 
```
cargo run --bin mcmc_ana_gibbs_2d4 --release
```
(yes, I hard coded the arguments in the source code, which should be corrected)
This program iteratively estimate the map and the noise power spectrum parameters.
