from base import hyperparams


class Hyperparams(hyperparams.Hyperparams):
    dtype = float
    # Trainer parameters:
    n_iterations = 100000

    show_visual_while_training = True
    train_generator_adv = True
    train_autoencoder = True

    train_batch_logits = True
    train_sample_logits = True

    start_tensorboard = True

    circular_bounds = False

    gen_iter_count = 40
    disc_iter_count = 80
    step_ratio = gen_iter_count, disc_iter_count

    disc_type = 'x'  # 'x' or 'z' or 'xz'

    # Dimension Parameters
    batch_size = 64
    seed_batch_size = 64

    logit_x_batch_size = 16
    logit_z_batch_size = 16

    # input_size = 2
    z_size = 100

    # Distribution params
    z_bounds = 4.
    cor = 0.6

    # Learning Parameters
    lr_autoencoder = 0.0001  # 0.0003
    lr_decoder = 0.0001  # 0.0003
    lr_disc = 0.0001  # 0.0003

    z_dist_type = 'normal'  # ['uniform', 'normal', 'sphere']

    model = 'gan'
    exp_name = 'mnist_exp_1'
    dataloader = 'mnist'  # check factory for values

    input_channel = 3
    input_height = 32
    input_width = 32
