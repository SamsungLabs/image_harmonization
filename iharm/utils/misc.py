import torch

from .log import logger


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


def save_checkpoint(net, checkpoints_path, epoch=None, prefix='', verbose=True, multi_gpu=False):
    if epoch is None:
        checkpoint_name = 'last_checkpoint.pth'
    else:
        checkpoint_name = f'{epoch:03d}.pth'

    if prefix:
        checkpoint_name = f'{prefix}_{checkpoint_name}'

    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)

    checkpoint_path = checkpoints_path / checkpoint_name
    if verbose:
        logger.info(f'Save checkpoint to {str(checkpoint_path)}')

    state_dict = net.module.state_dict() if multi_gpu else net.state_dict()
    torch.save(state_dict, str(checkpoint_path))


def load_weights(model, path_to_weights, verbose=False):
    if verbose:
        logger.info(f'Load checkpoint from path: {path_to_weights}')

    current_state_dict = model.state_dict()
    new_state_dict = torch.load(str(path_to_weights), map_location='cpu')
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)
