# README

This repository contains the code for the paper Composable Sparse Subnetworks via Maximum-Entropy Principle. It includes scripts and configuration files to reproduce all experiments presented in the paper, including submodule training, evaluation, merging and mode connectivity analysis.

## Running the Experiments
Below are the main commands used for training submodules, evaluating them, and analyzing merges.

### 1.Single-Class Evaluation
Run evaluation for a submodule trained to recognize class 0:
```bash
python frankenstein_sample_level.py single-evaluation-statistics \
  configs/setting_shallow_mlp_mnist.yaml 0 \
  -o ./out_dir \
  -s 0 -s 1 -s 2 -s 3 -s 4
```
### 2.Submodule Training (All Classes)
Train one submodule per class with pruning and multiple seeds:
```bash
python dataset_creation.py \
  --config configs/setting_shallow_mlp_mnist.yaml \
  --dataset_name_custom mnist \
  --lth_custom True --n_classes 1 --seed 0,1,2,3,4
```
### 3.Generate Tables and Plots
Automatically collects metrics and plots for section 2 experiments:
```bash
python results_tables.py -r 2 -d mnist --mod shallow_mlp
```
### 4.Pairwise Merge Evaluation
Evaluate the merge of submodules trained on disjoint class groups:
```bash
python frankenstein_sample_level.py single-comparison-statistics \
  configs/setting_shallow_mlp_mnist.yaml \
  -c1 0 -c1 1 -c1 2 -c1 3 -c1 4 \
  -c2 5 -c2 6 -c2 7 -c2 8 -c2 9 \
  -o ./out_dir \
  -s 0 -s 1 -s 2 -s 3 -s 4
```
### 5.Complete Merge and Loss Barrier Analysis
Incrementally merge submodules and evaluate for mode connectivity:
```bash
python frankenstein_sample_level.py incremental-comparison-statistics \
  configs/setting_deep_mlp_mnist.yaml \
  -lb -l train -re mean -a 5 \
  -o ./out_dir/ \
  -s 0 -s 1 -s 2 -s 3 -s 4 \
  -c 0 -c 1 -c 2 -c 3 -c 4
```
#### Flags explained:
- `-lb` : enables loss barrier computation.
- `-l train`: computes loss on the training set (can also use test) for the loss barrier.
- `-a 5`: computes the barrier on 5 points in the path.

### 6.Plotting Loss Barrier Results
If the `-lb` flag was used to collect mode connectivity data, use the following command to generate the corresponding plots:
```bash
python lb_plots.py
```

## Configuration Files
All experimental hyperparameters and training settings are defined in YAML files under the `configs/` directory. These include model architecture, dataset, optimizer, pruning ratio, and temperature settings.
For full transparency and reproducibility, all used configuration files are provided in the code submission.