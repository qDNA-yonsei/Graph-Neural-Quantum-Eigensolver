# Barren Plateau

Paper mapping: barren plateau experiment comparing gradient variance for VQE, NN-VQE, and EGATE-NNVQE.

Files:
- `vqe_bp_gradient_variance.ipynb`: VQE random-parameter baseline.
- `nnvqe_bp_gradient_variance.ipynb`: NN-VQE gradient variance.
- `egate_nnvqe_bp_gradient_variance.ipynb`: EGATE-NNVQE gradient variance.
- `NNVQE_HEA_half_uni.py` and `EGATE.py`: helper code.

Notes:
- Notebook outputs are assumed to be empty; run cells to regenerate results.
- The notebooks use `n_list`, `d`, and `iter` to control qubit count, ansatz depth, and repeats.
- No external `.pt`, checkpoint, or pickle files are required.
