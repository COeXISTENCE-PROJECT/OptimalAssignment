# GenTTP: Generalised Travell Time Predictor
This is the original implementation of GenTTP.


<p align="center">
  <img width="350" height="400" src=./images/model.pdf>
</p>

## Requirements
- python 3
- see `renvironment.py`

## Training Commands

```
python train.py
```

## Inference Commands

```
python inference.py
```

## Datasets

- ```8k_grid``` - 8196 assignments from grid method
- ```4k_grid``` - subset of 4092 assignmnets from ```8k_grid```
- ```1k_grid``` - subset of 1024 assignments from ```8k_grid```
- ```1k_random```- 1024 assignments from random method
- ```1k_greedy``` - 1024 assignments from greedy method
- ```1k_dirichlet``` - 1024 assignments from greedy method
