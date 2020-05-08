import os.path as osp
from exp_context import ExperimentContext
from utils.decorators import classproperty


class Paths(object):
    @classproperty
    def exp_name(cls):
        return ExperimentContext.exp_name

    @classproperty
    def current_exp_dir(cls):
        return osp.join(experiments_base_dir, cls.exp_name)

    @classproperty
    def logs_base_dir(cls):
        return osp.join(experiments_base_dir, cls.exp_name, 'logs')

    @classproperty
    def weights_base_dir(cls):
        return osp.join(experiments_base_dir, cls.exp_name, 'weights')

    @classproperty
    def results_base_dir(cls):
        return osp.join(experiments_base_dir, cls.exp_name, 'results')

    @classproperty
    def exp_hyperparams_file(cls):
        return osp.join(experiments_base_dir, cls.exp_name, 'hyperparams.json')

    @classproperty
    def temp_dir(cls):
        return osp.join(cls.results_base_dir, '.temp')

    @classmethod
    def log_writer_path(cls, writer_name, model_name=None):
        model_name = model_name or ''
        return osp.join(cls.logs_base_dir, model_name, writer_name)

    @classmethod
    def weight_dir_path(cls, dir_name):
        return osp.join(cls.weights_base_dir, dir_name)

    @classproperty
    def all_weights_dir(cls):
        return osp.join(cls.weights_base_dir, 'all/')

    @classproperty
    def saved_weights_dir(cls):
        return osp.join(cls.weights_base_dir, 'saved/')

    def weights_dir_paths(cls):
        return {
            'all': cls.all_weights_dir,
            'saved': cls.saved_weights_dir,
        }

    @classmethod
    def get_result_path(cls, path, model_name=None):
        model_name = model_name or ''
        return osp.join(Paths.results_base_dir, model_name, path)

    @classmethod
    def get_temp_file_path(cls, path, model_name=None):
        model_name = model_name or ''
        return osp.join(Paths.temp_dir, model_name, path)

    @classmethod
    def get_params_dir_path(cls, dir_name, model_name=''):
        return osp.join(Paths.weights_base_dir, dir_name, model_name)

    @classmethod
    def get_saved_params_path(cls, dir_name, model_name, weight_label, iter_no):
        dir_path = cls.get_params_dir_path(dir_name)
        if model_name is None:
            return osp.join(dir_path, '%s-%d' % (weight_label, iter_no))
        if iter_no is None:
            return osp.join(dir_path, model_name, '%s' % (weight_label))
        else:
            return osp.join(dir_path, model_name, '%s-%d' % (weight_label, iter_no))


def exp_name():
    return ExperimentContext.exp_name


experiments_base_dir = '../experiments'
