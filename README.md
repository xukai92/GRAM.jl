# Source code for *Generative Ratio Matching Networks*

The repo is arranged as below

- `Manifest.toml`: environment file
- `Project.toml`: project dependencies
- `examples/`
  - `Hyper.toml`: hyper parameter settings
  - `demo.ipynb`: notebook to paly with the model interactively
  - `parallel_exps.jl`: run experiments in Section 3.4 parallelly
- `src/`
  - `RMMMDNets.jl`: helper functions
  - `anonymized.jl`: codes copied for anonymization purpose
  - `data.jl`: definition of data
  - `modules.jl`: modules including generator, discriminator and projector
  - `models.jl`: models including GAN, MMD-net and GRAM-net
- ~~`tf_logs.zip`: TensorBoard logs for Section 3.4~~

**NOTE**: GRAM-net is also referred `RMMMDNet` or `rmmmdnet` across the code.

**UPDATE**: As the [anonymous_github](https://github.com/tdurieux/anonymous_github/) service fails to support [Git LFS](https://git-lfs.github.com/). We now provide our TensorBoard logs via Google Drive anonymously. Please download them from [here](https://drive.google.com/file/d/11vBwqom3he2RxgBtOypeFJD1ut-URWE4/view?usp=sharing).

### How to run the code

1. Install [Julia](https://julialang.org/downloads/) and make `julia` available in your executable path.
2. Download the code in a location which we will refer as `GRAM_DIR`.
3. Start a Julia REPL by entering `julia` in your terminal.
    - Press `]` button to enter the package manager.
    - Input `dev $GRAM_DIR`.
        - This will install all the Julia dependencies for you and might take a while.
    - Activate the project environment by `activate $GRAM_DIR`.
    - Press `delete` or `backspace` to exit the package manager.
    - Input `using PyCall`.
        - Input `PyCall.Conda.add("matplotlib")` to install matplotlib.
    - Exit the REPL.
4. Do `julia --project=$GRAM_DIR $GRAM_DIR/examples/parallel_exps.jl`
    - This by default produce Figure 1 (and Figure 6 in the appendix).
    - To produce other plots, you need to edit `parallel_exps.jl` as below and run the same command.
        - For Figure 2, uncomment L60 and L61.
        - For Figure 7 in the appendix, uncomment L64.
    - This script by default using 9 cores to run experiments in parallel. If you want to use another number of cores, please change L2 of `parallel_exps.jl`. 

Our code by default logs all the training details in `$GRAM_DIR/logs`, for which you can view using TensorBoard. 