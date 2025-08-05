## Creating the MIMIC-III Cohort

1. Obtain access to the [MIMIC-III Dataset on PhysioNet](https://physionet.org/content/mimiciii/1.4/) and download the .csv files.
2. Clone the [MIMIC-benchmarks repository](https://github.com/YerevaNN/mimic3-benchmarks), and follow all of the steps in the `Building a benchmark` section.
3. Update the `mimic_benchmark_dir` variable in `notebooks/GetMortalityData.ipynb`, and run all cells in the notebook sequentially to generate the dataset. 
4. Copy the `mimic_mortality_tabular.csv` into the datasets folder.