from dataloaders.factory import DataLoaderFactory

import numpy as np
import os, argparse, json, logging
import matplotlib.pyplot as plt

import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')
########################
from exp_context import ExperimentContext

default_args_str = '-hp hyperparameters/cifar_10.py -d all -en gan_cifar -t -g 0'
# Argument Parsing
parser = argparse.ArgumentParser()

parser.add_argument('-g', '--gpu', default=1, help='index of the gpu to be used. default: 0')
parser.add_argument('-t', '--tensorboard', default=False, const=True, nargs='?',
                    help='Start Tensorboard with the experiment')
parser.add_argument('-r', '--resume', nargs='?', const=True, default=False,
                    help='if present, the training resumes from the latest step, '
                         'for custom step number, provide it as argument value')
parser.add_argument('-d', '--delete', nargs='+', default=[], choices=['logs', 'weights', 'results', 'all'],
                    help='delete the entities')
parser.add_argument('-w', '--weights', nargs='?', default='iter', choices=['iter', 'best_gen', 'best_pred'],
                    help='weight type to load if resume flag is provided. default: iter')
parser.add_argument('-hp', '--hyperparams', required=True, help='hyperparam class to use from HyperparamFactory')
parser.add_argument('-en', '--exp_name', default=None,
                    help='experiment name. if not provided, it is taken from Hyperparams')

args = parser.parse_args(default_args_str.split())

print(json.dumps(args.__dict__, indent=2))

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
resume_flag = args.resume is not False
ExperimentContext.set_context(args.hyperparams, args.exp_name)
H = ExperimentContext.Hyperparams  # type: Hyperparams
exp_name = ExperimentContext.exp_name

##########  Set Logging  ###########
logger = logging.getLogger(__name__)
LOG_FORMAT = "[{}: %(filename)s: %(lineno)3s] %(levelname)s: %(funcName)s(): %(message)s".format(
    ExperimentContext.exp_name)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

#### Clear Logs and Results based on the argument flags ####
from paths import Paths
from utils import bash_utils, model_utils

if 'all' in args.delete or 'logs' in args.delete or resume_flag is False:
    logger.warning('Deleting all results in {}...'.format(Paths.logs_base_dir))
    bash_utils.delete_recursive(Paths.logs_base_dir)
    print('')

if 'all' in args.delete or 'results' in args.delete:
    logger.warning('Deleting all results in {}...'.format(Paths.results_base_dir))
    bash_utils.delete_recursive(Paths.results_base_dir)
    print('')

##### Create required directories
model_utils.setup_dirs()
print('dirs created')
###################

#dl = DataLoaderFactory.get_dataloader(H.dataloader, H.batch_size, H.batch_size)

# import base.dataloader as DL
from models.mnist.gan_cifar import ImgGAN
from dataloaders.mnist import CifarDataLoader

mdl = CifarDataLoader(train_batch_size=H.batch_size, test_batch_size=H.batch_size)
x,_ = mdl.next_batch('train')

# train_data, test_data, train_labels, test_labels = mdl.get_data()

from trainer.image_trainer import ImgTrainer
from configs import Config

model = ImgGAN.create_from_hyperparams(name="fashion-gan-2", hyperparams=H)

# model.load_params(dir_name='iter', weight_label='iter', iter_no=99000)
# logger.info("model safely loaded")

trainer = ImgTrainer(model=model, data_loader=mdl, train_config=Config.default_train_config, hyperparams=H)

for i in range(20):
    trainer.train(10000)
    model.save_params(dir_name='iter', weight_label='all', iter_no=i)
