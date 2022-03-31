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
    "batch_size": 1,
    "lr_scheduler": "plateau",
    "scheduler_gamma": 0.7,
    "scheduler_patience": 5,
    "scheduler_max": 0.1,
    "optimizer": "adam",
    "momentum": 0.7,
    "gpu": 1,
    "accumulate": 7,
    "checkpoint": "",
    "sweep": False,
    "sweepid": "",
    "sweepruns": 1,
    "downsampler": "", #"", stride, maxpool
    "hiddensize": 64,
    "lstmlayers": 4,
    'is_training': 1, 
    'img_width': 64, 
    'img_channel': 1, 
    'model_name': '',
    'pretrained_model': '', 
    'num_hidden': '64 64 64 64', 
    'filter_size': 5, 
    'dataset_slice': -1, 
    'internal_valid': False,
    'validation': '',
    'stride': 1, 'patch_size': 4, 'layer_norm': 1, 'decouple_beta': 0.1, 'reverse_scheduled_sampling': 0, 'r_sampling_step_1': 25000, 'r_sampling_step_2': 
50000, 'r_exp_alpha': 5000, 'scheduled_sampling': 1, 'sampling_stop_iter': 50000, 'sampling_start_value': 1.0, 'sampling_changing_rate': 2e-05, 'reverse_input': 1, 'max_iterations': 80000, 'display_interval': 100, 
'test_interval': 5000, 'snapshot_interval': 5000, 'num_save_samples': 10, 'visual': 0, 'visual_path': './decoupling_visual', 'injection_action': 'concat', 'conv_on_input': 0, 'res_on_conv': 0, 'num_action_ch': 4,
'embedding_dim': 256, 'n_codes': 2048, 'n_hiddens': 240, 'n_res_layers': 4, 'downsample': (4, 4, 4)
}

# default_config = dict([(k.replace('_', ''), v) for k, v in default_config.items()])

arg_names = ('type', 'default', 'dest')

default_arguments = dict([(name, [f"--{name.replace('_', '')}", type(value), value, name.replace('_', '')]) for name, value in default_config.items()])
default_arguments['separate'][1] = str
default_arguments['separate'][2] = "32 64 128 1"
default_arguments['inner_size'][1] = str
default_arguments['inner_size'][2] = "8 32 64"
default_arguments['num_hidden'][1] = str
default_arguments['num_hidden'][2] = "64,64,64,64"


def add_arguments(parser : argparse.ArgumentParser):
    for _, v in default_arguments.items():
        parser.add_argument(v[0], **dict(zip(arg_names, v[1:])))