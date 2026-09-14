# CS6886 Assignment 4: ANN-to-SNN conversion

## Environment setup

This project needs `torch`, `torchvision`, `numpy`, `matplotlib`, `jupyter`, and a local install of `spikingjelly` (a vendored fork in this repo, under `spikingjelly/`).

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate cs6886_a3
```

Then select the `cs6886_a3` kernel in `A4.ipynb`.

If you already have a working environment with these packages, including `spikingjelly` installed from `./spikingjelly/setup.py`, you can skip this step.

## What the notebook does

`A4.ipynb` walks through converting a small CNN, trained on MNIST, into a spiking neural network (SNN), then measures the tradeoffs.

- **Task 1-2**: install prerequisites, count MAC operations in each conv layer of the original CNN.
- **Task 3**: sweep four ANN-to-SNN conversion presets (`max`, `99.9%`, `1/2 max`, `1/4 max`) and pick the one that reaches its own peak accuracy in the fewest timesteps.
- **Task 4**: compare accuracy over timesteps for the original CNN against 1, 2, and 3 converted conv blocks.
- **Task 5**: measure the average number of spikes each layer fires per image, for a fully converted SNN.
- **Task 6**: find the best accuracy achievable at 90% or higher energy savings, using a compute-only cost model (MACs vs. accumulate operations). Answer: fully converting the network is required to reach 90% savings, at a real accuracy cost.
- **Task 7**: find the best energy savings achievable while keeping accuracy loss under 5%. Answer: partial conversion (2 of 3 blocks) beats full conversion here, since it needs fewer timesteps to stay accurate.
- **Task 8**: calculate the memory-write energy cost of each layer, for a model where writing to memory costs a fixed amount per 64 bytes.
- **Task 9**: repeat Task 7's question, but with the memory-aware cost model instead of the compute-only one. Answer: no configuration actually saves energy once memory writes are counted — the best case only breaks even with the original CNN.

Tasks 6, 7, and 9 search many `(blocks converted, timesteps, preset)` combinations in parallel (using separate OS processes) to find each answer, rather than picking values by hand.

## AI usage

Claude Code was used to help complete this assignment:
- Wrote the parallel search functions (`search_conversion_configs`, `search_conversion_configs_memory`) in `cnn_mnist.py`.
- Reviewed my code logic for bugs and was used to polish the code structuring.
- Wrote code comments
- Was used to iteratively reduce the search space for optimal results for Q6,7,9

## GitHub URL

This repo is publicly available at: https://github.com/starkAhmed43/CS6886_A3.git
