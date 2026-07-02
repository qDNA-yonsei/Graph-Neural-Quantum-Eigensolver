# 1D XXZ+X

Paper mapping: generalization experiment for 1D XXZ+X.

Run order:
1. `egate_data.ipynb`
2. `analytic_8qubit.ipynb`
3. `nnvqe_*.ipynb` and `egate_nnvqe_*.ipynb`
4. `random_parameter_test_8qubit.ipynb` for the appendix random parameter baseline table

`analytic_8qubit.ipynb` generates train/test analytical `.pt` files used by the training notebooks.

Set `seed` in the notebooks for repeated seed runs.
