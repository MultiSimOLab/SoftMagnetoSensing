# Neural-network surrogate for piezoresistive binary sensors

Code and data for the neural-network surrogate, sensitivity analysis and inverse design reported in

> J. Crespo-Miguel, R. Ortigosa, M. L. Lopez-Donaire, J. Pernas-Sanchez, J. Martinez-Frutos, D. Garcia-Gonzalez. *Piezoresistive Binary Sensors Through Residual Magnetisation*.

The finite element simulations that generate the training database are in [`../FE_simulations`](../FE_simulations) (HyperFEM.jl).

The surrogate maps seven design variables of the multi-domain sensor

| symbol | meaning | unit |
|---|---|---|
| `w` | magnetic-domain length | mm |
| `e` | magnetic-domain width | mm |
| `phi_HM` | hard-magnetic particle volume fraction | vol% |
| `d_BC` | clamp offset (multiple of `w`) | - |
| `W`, `h` | device width and length | mm |
| `c` | conductive-pathway width | mm |

onto the two recovery displacements `u_On` and `u_Union` (mm) computed by the FE model. The trigger strain is `eps = u / (2 w (1 + d_BC))`.

## Contents

```
data/fe_database.bson     FE database, 452 simulations (7 inputs, 2 outputs)
data/fe_database.csv      same data as plain text
models/surrogate_L4_N8.bson
                          released network: weights, scaling constants,
                          loss history and training indices
src/Surrogate.jl          curation, scaling, network, training, Sobol'
                          indices, GP baseline, inverse design
scripts/example.jl        quick demonstration (prediction + inverse design)
scripts/train.jl          training from scratch
scripts/si_analysis.jl    numbers and data of Supplementary Section S3
scripts/fig4_sensitivity.jl
                          sensitivity analysis and maps of Fig. 4c-e
plotting/si_figures.py    Supplementary Figs. S3-S7
plotting/fig4_panels.py   Fig. 4c-e
results/                  expected outputs (JSON and text logs)
figures/                  expected figures
```

## Requirements

- Julia 1.10 or later (tested with 1.12.6). All Julia dependencies are pinned in `Project.toml` / `Manifest.toml` (Flux 0.16, Zygote 0.7, BSON, JSON, Distributions, GaussianProcesses).
- Python 3 with `numpy` and `matplotlib`, only for the figures.

No GPU is needed. Everything runs on a single CPU core.

## Installation

```
git clone https://github.com/MultiSimOLab/SoftMagnetoSensing
cd SoftMagnetoSensing/NN_surrogate
julia --project=. -e "using Pkg; Pkg.instantiate()"
pip install -r requirements.txt
```

Installing and precompiling the Julia packages takes 5-10 min the first time.

## Usage

All commands are run from this folder (`NN_surrogate`).

**Quick check** (less than 1 min after precompilation):

```
julia --project=. scripts/example.jl
```

Predicts the 49 held-out FE designs (expected: R2 = 0.99830 for `u_On` and 0.99935 for `u_Union`, RMSE 22.5 and 29.8 um), evaluates a new design and solves one inverse-design problem. The full expected output is in `results/example_output.txt`.

**Supplementary analysis** (about 1.5 min):

```
julia --project=. scripts/si_analysis.jl
python3 plotting/si_figures.py
```

Writes `results/si_nn_data.json` and `figures/{curation,loss,parity,sobol,sobol_family}.pdf`.

**Fig. 4 sensitivity analysis** (about 1.5 min):

```
julia --project=. scripts/fig4_sensitivity.jl
python3 plotting/fig4_panels.py
```

Writes `results/fig4_sensitivity_data.json` and `figures/fig4_cd.pdf`, `figures/fig4_e.pdf`.

**Training from scratch** (about 2-3 min for 1e5 epochs):

```
julia --project=. scripts/train.jl [epochs] [seed]
```

Saves `models/retrained_L4_N8.bson`. A new run draws its own training split and initialisation, so the weights differ from the released ones while the accuracy is equivalent: with the default seed we obtain a held-out R2 of 0.99863 (released network: 0.99886) and held-out RMSE of 24.2 / 36.1 um (log in `results/train_output.txt`). The numbers in the paper are obtained from the released network, whose split is stored in the model file.

## Method summary

1. **Curation.** Designs that do not reconnect within the 15 mm stroke are removed (452 -> 450), then outliers in the output space are removed by a Mahalanobis filter at chi-square level 0.9 (450 -> 409).
2. **Split.** 360 designs for training, 49 held out; random split, stored with the model. No hyperparameter is tuned on the held-out set.
3. **Scaling.** Inputs mapped to [-1, 1] and outputs standardised, with constants computed on the training split only.
4. **Network.** 4 hidden layers of 8 tanh units, linear output, 298 parameters.
5. **Training.** Relative Frobenius-norm loss, Adam (learning rate 1e-2) with L2 penalty 1e-6, mini-batches of 250, 1e5 epochs, best iterate on the training loss.
6. **Baseline.** Gaussian process per output (ARD Matern 5/2) on the same split.
7. **Sensitivity.** Sobol' indices by pick-and-freeze Monte Carlo (Saltelli and Jansen estimators) with bootstrap intervals; all random stages are seeded (`SEED = 20250820`).
8. **Inverse design.** Gradient-based search (Zygote + Adam with box projection) on the trigger strains, with the constraint `w (1 + d_BC) <= h`.

## Data format

`data/fe_database.csv` has one row per FE simulation: `simulation_id`, the seven inputs and the two outputs. In `simulation_id`, the field `c_` is the particle volume fraction `phi_HM` and `αgrip` is `d_BC`. In the BSON file, `:data` is a 452 x 9 matrix with the same column order.

## License

MIT, see [`../LICENSE`](../LICENSE).
