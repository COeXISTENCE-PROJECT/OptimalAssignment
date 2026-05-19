# GenTTP: Generalised Travell Time Predictor
This is the original implementation of GenTTP.


<p align="center">
  <img width="350" height="400" src=./images/model.pdf>
</p>

Minimal training code for the GenTTP model used in the paper.

## Requirements

The repository should contain at least:

- `train.py`
- `engine.py`
- `utilities.py`
- a directory with flow files, passed as `--q_dir`
- a directory with assignment files, passed as `--a_dir`
- an adjacency matrix CSV, passed as `--adjdata`

The flow and assignment directories should contain matching files. The adjacency matrix must match `--num_nodes`.

Install the Python dependencies used by the training script, including PyTorch, NumPy, and pandas.

## Training

Training can be launched directly:

```bash
python train.py \
  --device cpu \
  --q_dir /path/to/flows \
  --a_dir /path/to/assignments \
  --adjdata /path/to/adjacency_matrix.csv \
  --save_dir /path/to/outputs \
  --exp_name genttp_run \
  --seed 42 \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --seq_length_q 15 \
  --seq_length_a 30 \
  --seq_length_y 1 \
  --num_nodes 195 \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --dropout 0.1 \
  --num_workers 4 \
  --sequence_model lstm \
  --fuse_method attention \
  --gcn_bool
```

## Outputs

Each run is saved under:

```text
<save_dir>/<exp_name>/
```

The directory contains:

- `best_model.pth` -- checkpoint selected by validation MAE
- `final_model.pth` -- final model checkpoint
- `training_metrics.csv` -- per-epoch training and validation metrics
- `learning_curves.png` -- training curves

## Datasets

- ```8k_grid``` - 8196 assignments from grid method
- ```4k_grid``` - subset of 4092 assignmnets from ```8k_grid```
- ```1k_grid``` - subset of 1024 assignments from ```8k_grid```
- ```1k_random```- 1024 assignments from random method
- ```1k_greedy``` - 1024 assignments from greedy method
- ```1k_dirichlet``` - 1024 assignments from greedy method


## Citation

If this repository is useful for your work, please cite the corresponding paper.

