# EGATE-NNVQE paper code

Representative notebooks and helper scripts for "Improving Generalization and Trainability of Quantum Eigensolvers via Graph Neural Encoding"

## Folders
- `01_1D_XXZ`: 1D XXZ generalization code.
- `02_1D_XXZ_X`: 1D XXZ+X generalization code.
- `03_2D_XXZ`: 3x3 2D XXZ generalization code.
- `04_2D_XYZ`: 3x3 2D XYZ generalization code.
- `05_2D_XXZ_3x4`: appendix 3x4 2D XXZ generalization code.
- `06_2D_XYZ_3x4`: appendix 3x4 2D XYZ generalization code.
- `07_SKQD`: SKQD initializer experiments for the same Hamiltonian families.
- `08_BP`: barren plateau gradient variance experiments.

## Notes
- Notebook outputs are assumed to be empty; run cells to regenerate results.
- Data files, checkpoints, pickle results, and generated figures are not included.
- Change `seed` in notebooks for seed repeats.
- Change `shots` in SKQD notebooks for shot sweeps.
- BP notebooks generate random Hamiltonians internally.
