import os
import torch
import importlib
import pkgutil  
from utils.example_task_policy import ensure_example_task_backbone_supported

# Just put your model files under models/ folder
# e.g., models/Transformer.py, models/LSTM.py, etc.
# All models will be automatically detected and can be used by specifying their names.

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.final_train_epoch = None
        
        # -------------------------------------------------------
        #  Automatically generate model map
        # -------------------------------------------------------
        model_map = self._scan_models_directory()
        model_name = str(getattr(self.args, 'model', '') or '').strip()
        if model_name and model_name not in model_map:
            raise NotImplementedError(f"Model [{model_name}] not found in 'models' directory.")

        # Gate experiment launches by the recipe combinations defined under examples/.
        ensure_example_task_backbone_supported(
            model_name,
            getattr(self.args, 'task_name', ''),
        )

        # Use smart dictionary
        self.model_dict = LazyModelDict(model_map)

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _scan_models_directory(self):
        """
        Automatically scan all .py files in the models folder
        """
        model_map = {}
        models_dir = 'models'

        # Iterate through all files in 'models' directory
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                # Ignore __init__.py and non-.py files
                if filename.endswith('.py') and filename != '__init__.py':
                    # Remove .py extension to get module name
                    module_name = filename[:-3]
                    
                    # Build full import path
                    full_path = f"{models_dir}.{module_name}"
                    
                    # loading dict: {'Transformer': 'models.Transformer'}
                    model_map[module_name] = full_path
        
        return model_map

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _results_folder_name(self, setting):
        results_id = str(getattr(self.args, 'results_id', '') or '').strip()
        if not results_id:
            return setting

        task_name = str(getattr(self.args, 'task_name', '') or '').strip() or 'task'
        model_id = str(getattr(self.args, 'model_id', '') or '').strip() or 'model'

        for sep in filter(None, {os.sep, os.altsep}):
            task_name = task_name.replace(sep, '_')
            model_id = model_id.replace(sep, '_')
            results_id = results_id.replace(sep, '_')

        return f'{task_name}_{model_id}_{results_id}'

    def _results_folder_path(self, setting):
        return os.path.join('.', 'results', self._results_folder_name(setting))

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class LazyModelDict(dict):
    """
    Smart Lazy-Loading Dictionary
    """
    def __init__(self, model_map):
        self.model_map = model_map
        super().__init__()

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        
        if key not in self.model_map:
            raise NotImplementedError(f"Model [{key}] not found in 'models' directory.")
            
        module_path = self.model_map[key]
        try:
            print(f"🚀 Lazy Loading: {key} ...") 
            module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"❌ Error: Failed to import model [{key}]. Dependencies missing?")
            raise e

        # Try to find the model class
        if hasattr(module, 'Model'):
            model_class = module.Model
        elif hasattr(module, key):
            model_class = getattr(module, key)
        else:
            raise AttributeError(f"Module {module_path} has no class 'Model' or '{key}'")

        self[key] = model_class
        return model_class
