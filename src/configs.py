from collections import namedtuple

import torch as tr

TrainConfig = namedtuple(
    'TrainConfig',
    'n_step_tboard_log '
    'n_step_console_log '
    'n_step_validation '
    'n_step_save_params '
    'n_step_visualize'
)

class Config:
    use_gpu = tr.cuda.is_available()

    dtype = tr.float32

    base_port = 8001

    default_train_config = TrainConfig(
        n_step_tboard_log=50,
        n_step_console_log=-1,
        n_step_validation=100,
        n_step_save_params=-1,
        n_step_visualize=200
    )
