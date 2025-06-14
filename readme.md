# Collection of benchmarking dataset for multimarginal trajectory inference

This repo provides a collection of benchmark datasets used in a sequence of papers targeting multimarginal trajectory and SDE inference from snapshot data [Shen et al. 2025](https://arxiv.org/abs/2408.06277), [Berlinghieri et al. 2025](https://arxiv.org/abs/2505.16082). 

There are in general three levels of realisticity: 1) fully synthetic, where we define a model and generate the data; 2) semi-synthetic, where we use real velocity fields but simulate particles; and 3) real, where we do not know the model and did not simulate the data. 

We provide the data, the method to generate the data, citations, and license information. Please consider citing the original work where the data was initially collected, generated, and/or introduced. 

The repository generally requires `torchsde`, and the data are in the shape `[snapshots, batch, dimensions]`. Consider changing the code if a different sampling frequency and/or sample size is desired. The data does have the same batch size for each snapshot and the snapshots are equally spaced, but this is not a required behavior. 

## Fully synthetic

### Lotka Volterra system 
This is a classic dynamical system describing predator-prey dynamics. To generate the data, follow

```
cd model
python make_data.py LV
```
Consider citing [Shen et al. 2025](https://arxiv.org/abs/2408.06277). The data is licensed with this repository. 

Data will be saved in `asset` as `LV_data.npz`. There are three keys: `N_steps`, the number of snapshots, and `Xs`, the data of size `[snapshots, batch, dimensions]`.

### Repressilator system with mRNA only 
The Repressilator system is a minimal biological clock. For more biological information, we recommend reading [this tutorial](https://biocircuits.github.io/chapters/09_repressilator.html). Note that this model is called a protein model in the tutorial because they assumed measurements were done with optics and thus measured protein. Usually, the mRNA-only model will look similar. To generate the data with the mRNA-only model: 

```
cd model
python make_data.py Repressilatormrna
```


Consider citing [Shen et al. 2025](https://arxiv.org/abs/2408.06277) and [Elowitz and Leibler 2000](https://www.nature.com/articles/35002125). The data is licensed with this repository. 

### Repressilator system with mRNA and protein
This model also includes protein in the reaction network, see [this](https://biocircuits.github.io/chapters/09_repressilator.html#Including-mRNA-in-the-model-provides-additional-insights). One can derive the mRNA-only model by assuming the protein quickly reaches equilibrium. 

To generate the data with the mRNA and protein model:

```
cd model
python make_data.py Repressilatormrnaprotein
```


Consider citing [Berlinghieri et al. 2025](https://arxiv.org/abs/2505.16082) and [Elowitz and Leibler 2000](https://www.nature.com/articles/35002125). The data is licensed with this repository. 

## Semi-synthetic

### Gulf of Mexico vortex

The data is from high-resolution (1 km) bathymetry data from a HYbrid Coordinate Ocean Model (HYCOM) reanalysis [Panagiotis 2014](https://doi.org/10.7266/N7X63JZ5). This dataset was released by the US Department of Defense and is thus in the public domain. The dataset provides hourly ocean current velocity fields for the region extending from 98E to 77E in longitude and from 18N to 32N in latitude, covering every day since January 1st, 2001. We focus on a specific time point, June 1st, 2024, at 5 PM at surface level, and a particular spatial region where a vortex is observed.

To generate the data:

```
cd model
python make_data.py GoM
```
The code will read HYCOM data from `asset` and simulate particles. 

Consider citing [Panagiotis 2014](https://doi.org/10.7266/N7X63JZ5) and [Shen et al. 2025](https://arxiv.org/abs/2408.06277). HYCOM data is funded by the US federal government and thus in the public domain. 

## Real data
The data is a real single-cell RNA-sequencing dataset that tracks T cell–mediated immune activation in peripheral blood mononuclear cells (PBMCs) [Jiang et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10153191/). They recorded gene-expression profiles every 30 minutes for 30 hours. We use the 41 snapshots collected between 0 h and 20 h—prior to the onset of steady state; we take 20 alternating snapshots (at integer hours) for training and the remaining 21 for validation.  
We use a 30-dimensional projection ("gene program") of the original measurements.

The original program data is in `asset/pbmc_timecourse.csv`. The first column is cell ID, the second to third-from-last are cell programs, while the last two are time collected and cell type annotations. A slightly cleaned and subsampled version is given in `asset/processed_pbmc_data_sub500_every_2_until20.npz`. 

Consider citing [Jiang et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10153191/) and [Berlinghieri et al. 2025](https://arxiv.org/abs/2505.16082). The data is under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) with [Jiang et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10153191/). 

## Contribution
We welcome contributions. Please start a PR and include a description of the data, original citation, and license.








