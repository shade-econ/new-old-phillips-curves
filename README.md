# New Pricing Models, Same Old Phillips Curves?
This repository shows in detail how to obtain the results from Section 6 of ["New Pricing Models, Same Old Phillips Curves?" (Auclert, Rigato, Rognlie, Straub 2023)](http://mattrognlie.com/new_old_phillips_curves.pdf).

The `section6.ipynb` Jupyter notebook, which you can [visualize on GitHub here](https://github.com/shade-econ/new-old-phillips-curves/blob/main/section6.ipynb),
goes through every step of the computation. It:
- Starts with raw price change data in `israeli_data_cleaned.csv`, derived from data in [Bonomo et al. (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3590402).
- Infers the generalized hazard function $\Lambda$ and variance of idiosyncratic shocks $\sigma_\epsilon$ that best fit the distribution and frequency of price changes.
- Uses $\Lambda$ and $\sigma_\epsilon$ to calculate the expected price gap functions $E^t(x)$.
- Implements the formula in Proposition 6 of the paper to obtain the pass-through matrix from aggregate nominal marginal costs to prices, $\Psi$.
- Converts this $\Psi$ to the generalized Phillips curve matrix $\mathbf{K}$.

The brief `utils.py` module contains a few helper functions that are used in the notebook.

If you want to play around with the notebook yourself, you can [download a zip of the repository here](https://github.com/shade-econ/new-old-phillips-curves/archive/refs/heads/main.zip).

In addition to Python and Jupyter notebook, our code requires the NumPy, SciPy (version 1.7.0+), Pandas, and Matplotlib packages. A good way to obtain these is to install the [latest Anaconda distribution](https://www.anaconda.com/distribution/).
