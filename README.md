# Source code for *Generative Ratio Matching Networks*

The repo is arranged as below

- `Manifest.toml`: environment file
- `Project.toml`: project dependencies
- `examples/`
  - `Hyper.toml`: hyper parameter settings
  - `demo.ipynb`: notebook to paly with the model interactively
  - `parallel_exps.jl`: run experiments in Section 3.4 parallely
- `src/`
  - `RMMMDNets.jl`: helper functions
  - `anonymized.jl`: codes copied for anonymization purpose
  - `data.jl`: definition of data
  - `modules.jl`: modules including generator, discriminator and projector
  - `models.jl`: models including GAN, MMD-net and GRAM-net

**NOTE**: GRAM-net is called `RMMMDNet` or `rmmmdnet` across the code.

## How to run the code

1. Install [Julia](https://julialang.org/downloads/) and make `julia` avaiable in your executable path.
2. Download the code in a location which we will refer as `GRAM_DIR`.
3. Start a Julia REPL by entering `julia` in your terminal.
  - Press `]` button to enter the package manager
  - Input `add $GRAM_DIR`
    - This will install all the Julia dependencies for you and might take a while.
  - Press `delete` or `backspace` to exit the package manager
  - Input `using PyCall`
    - Input `PyCall.Conda.add("matplotlib")` to install matplotlib.
    - Input `PyCall.Conda.add("tensorboard")` to install TensorBoard.
  - Exit the REPL.
4. Do `julia --project=$GRAM_DIR $GRAM_DIR/examples/parallel_exps.jl`
  - This by default pro