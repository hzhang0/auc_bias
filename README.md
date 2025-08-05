# A Closer Look at AUROC and AUPRC under Class Imbalance

## Paper
If you use this code in your research, please cite the following paper:

```
@article{mcdermott2024closer,
  title={A closer look at auroc and auprc under class imbalance},
  author={McDermott, Matthew and Zhang, Haoran and Hansen, Lasse and Angelotti, Giovanni and Gallifant, Jack},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={44102--44163},
  year={2024}
}
```

## To replicate the experiments in the paper:

### Setting Up

Run the following commands to clone this repo and create the Conda environment:

```
git clone git@github.com:hzhang0/auc_bias.git
cd auc_bias
conda env create -f environment.yml
conda activate auc_bias
```


### Synthetic Experiments

To reproduce the experiments on synthetic data (Section 3.1 of the paper), run the `notebooks/synthetic_exps.ipynb` notebook top to bottom.


### Training a Single Model
To train a single model, call `train.py` with the appropriate arguments, for example:

```
python -m auc_biases.train \
    --output_dir /output/dir \
    --dataset adult \
    --algorithm xgb \
    --balance_groups \
    --attribute 0 \
    --higher_prev_group_weight 3
```

To obtain the `mimic` dataset, see instructions [here](./ProcessingMIMIC.md). The other three datasets are included and/or downloaded automatically.


### Training a Grid of Models

To reproduce the experiments in the paper which involve training a grid of models using different hyperparameters, use `sweep.py` as follows:

```
python sweep.py launch \
    --experiment {experiment_name} \
    --output_dir {output_root} \
    --command_launcher {launcher} 
```

where:
- `experiment_name` corresponds to experiments defined as classes in `experiments.py`
- `output_root` is a directory where experimental results will be stored.
- `launcher` is a string corresponding to a launcher defined in `launchers.py` (i.e. `slurm` or `local`).

The experiment `vary_group_weight_with_seeds` corresponds to Figure 3. We have also uploaded the results of this experiment [here](https://www.dropbox.com/scl/fi/o5ye4d2lh02k39gsm57ze/vary_group_weight_with_seeds_res.pkl?rlkey=en20kiimzc8nunfkj4ajx8aj3&dl=0). You can download this pickle file and place it in the `notebooks` folder before continuing to the next step.

### Aggregating Results

After an experiment has finished running, to create Figures 3, 7, and 8, run `notebooks/agg_results.ipynb`
