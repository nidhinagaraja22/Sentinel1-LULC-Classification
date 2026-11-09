"""Run this first in Data Lab. No extra packages are required."""
import importlib.util
import platform
import shutil
import sys

print('Python:', sys.version.split()[0])
print('Operating system:', platform.system(), platform.machine())
print('Free disk space: {:.1f} GB'.format(shutil.disk_usage('.').free / 10**9))
for name in ('numpy', 'torch', 'rasterio', 'streamlit'):
    print(f'{name}:', 'installed' if importlib.util.find_spec(name) else 'not installed')
if importlib.util.find_spec('torch'):
    import torch
    print('PyTorch version:', torch.__version__)
    print('CUDA GPU available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
print('Environment check complete. Missing packages are normal at this stage.')
