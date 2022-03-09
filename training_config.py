import argparse
default_config = {
    "separate": (32, 64, 128, 1),
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
    "convert": False,
    "batch_size": 1
}


arg_names = ('type', 'default', 'dest')

default_arguments = dict([(name, [f"--{name.replace('_', '')}", type(value), value, name.replace('_', '')]) for name, value in default_config.items()])
default_arguments['separate'][1] = str
default_arguments['separate'][2] = "32 64 128 1"
default_arguments['inner_size'][1] = str
default_arguments['inner_size'][2] = "8 32 64"


def add_arguments(parser : argparse.ArgumentParser):
    for _, v in default_arguments.items():
        parser.add_argument(v[0], **dict(zip(arg_names, v[1:])))
        