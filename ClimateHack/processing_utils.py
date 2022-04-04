from einops import rearrange
import torch

# Processing utils for TrajGru Train flow.
# preprocessing and postprocessing allow for normalizing features per patch or globally
def preprocessing(config, batch_features):
    MEAN = 299.17117
    STD = 146.06215
    mi1 = None
    ma1 = None
    if config['normalize']:
        if config['localnorm']:
            MEAN = torch.mean(batch_features).detach()
            STD = torch.std(batch_features).detach()
        if config['normalize'] == 'standardize':
            batch_features -= MEAN
            batch_features /= STD
        elif config['normalize'] == 'minmax':
            mi1 = batch_features.min().detach()
            ma1 = batch_features.max().detach()
            batch_features -= mi1
            batch_features /= (ma1 - mi1)
            batch_features *= 2
            batch_features -= 1
        elif config['normalize'] == 'classic':
            batch_features /= 1023

    return batch_features, {'MEAN':MEAN, 'STD':STD, 'mi1':mi1, 'ma1':ma1}

def postprocessing(config, predictions, extra):
    MEAN = extra['MEAN']
    STD = extra['STD']
    mi1 = extra['mi1']
    ma1 = extra['ma1']
    if config['convert']:
        predictions = rearrange(predictions, 'b t c h w -> b (t c) h w')
    if config['normalize']:
        if config['outputstd']:
            MEAN = config['output_mean']
            STD = config['output_std']
        if config['normalize'] == 'standardize':
            predictions *= STD
            predictions += MEAN
        elif config['normalize'] == 'minmax':
            predictions += 1
            predictions /= 2
            predictions *= (ma1 - mi1)
            predictions += mi1
        elif config['normalize'] == 'classic':
            predictions = torch.clamp(predictions, 0, 1)
            predictions *= 1023
    return predictions