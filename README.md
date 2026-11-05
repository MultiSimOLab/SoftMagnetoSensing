# SoftMagnetoSensing

Simulation code and data of

> J. Crespo-Miguel, R. Ortigosa, M. L. Lopez-Donaire, J. Pernas-Sanchez, J. Martinez-Frutos, D. Garcia-Gonzalez.
> *Piezoresistive Binary Sensors Through Residual Magnetisation.* Submitted to Nature Communications.

The sensor is a pair of hard-magnetic grippers with residual magnetisation facing each other across a
gap. Under compression the magnetic attraction closes the gap and switches a conductive pathway on and
off: a binary, piezoresistive response set by the geometry and the particle concentration of the device.

The repository has two independent parts, each with its own Julia environment and README:

| Folder | Content |
|---|---|
| [`FE_simulations/`](FE_simulations/) | Finite element model of the sensor (magneto-mechanics with explicit air, [HyperFEM](https://github.com/MultiSimOLab/HyperFEM.jl) on [Gridap](https://github.com/gridap/Gridap.jl)) and the parametric study over the gripper width, the gap and the particle concentration that generates the database of the paper |
| [`NN_surrogate/`](NN_surrogate/) | Neural-network surrogate trained on the finite element database, Sobol' sensitivity analysis and gradient-based inverse design (Fig. 4 and Supplementary Section S3) |

## Requirements

- Julia ≥ 1.12 for `FE_simulations` (HyperFEM 0.2, HyperFEMKernels 0.1, Gridap 0.19.9) and Julia ≥ 1.10 for
  `NN_surrogate` (Flux 0.16). Every dependency is a registered Julia package, pinned in the `Manifest.toml`
  of each folder: `julia --project=. -e 'using Pkg; Pkg.instantiate()'` inside the folder installs it.
- Python 3 with `numpy` and `matplotlib`, only for the figures of `NN_surrogate`.

## Citation

Please cite the paper above. The finite element simulations rely on HyperFEM:

C. Perez-Garcia, R. Ortigosa, J. Martínez-Frutos, D. Garcia-Gonzalez. *Topology and material optimization in
ultra-soft magnetoactive structures: making advantage of residual anisotropies.* Adv. Mater. (2025): e18489.
https://doi.org/10.1002/adma.202518489

## License

MIT, see [`LICENSE`](LICENSE).

## Contact

Jesús Martínez-Frutos (jesus.martinez@upct.es), MultiSimO Lab, Universidad Politécnica de Cartagena.
