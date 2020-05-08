
import importlib
import base.hyperparams as base_hp

class HyperparamsFactory:
    class_type = base_hp.Hyperparams

    @classmethod
    def get_hyperparams(cls, module_name):
        # type: (str) -> dcgan_2d.Hyperparams
        # return cls.__dict[name]

        print('importing hyperparams %s' % module_name)
        module = importlib.import_module(module_name)
        Hyperparams = module.Hyperparams
        return Hyperparams

