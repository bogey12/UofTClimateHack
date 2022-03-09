
default_config = {
    "separate": (32, 64, 128),
    "n_layers": 1,
    "dropout": 0.2,
    "swap": False,
    "lr": 1e-3,
    "normalize": 'standardize', 
    "local_norm": True,
    "output_mean": 0,
    "output_std": 0,
    "opt_flow": False,
    "inputs": 12,
    "outputs": 24,
    "in_opt_flow": False,
    "criterion": 'msssim',
    "weight_decay": 1e-8,
    "epochs": 200,
    "dataset": '/datastores/ds-total/ds_total.npz',
    "patience": 20,
    "inner_size": (8, 32, 64),
    "convert": False
}