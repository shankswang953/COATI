# Synthetic data

- [Balanced Gaussian bump](balancedGaussianbump.ipynb)
- [Unbalanced Gaussian bump](unbalancedGaussianbump.ipynb)
- [Cy comparison](cyComparison.ipynb)
- [BranchSync](BranchSync.ipynb)

The Gaussian tutorials use the same committed AnnData and frozen cross-modal map.
The unbalanced Gaussian example changes the terminal mass target and learns growth
in addition to the trajectory. BranchSync directly generates its five paired
ToySplit snapshots and refits a time-conditioned map before balanced training.
