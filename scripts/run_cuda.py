"""Optional reuse of an existing compatible CUDA torch installation, without changing it."""
import argparse
import runpy
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--torch-site',required=True); p.add_argument('script'); p.add_argument('args',nargs=argparse.REMAINDER); a=p.parse_args()
    sys.path.insert(0,a.torch_site)
    import torch
    sys.path.pop(0)
    if not torch.cuda.is_available(): raise RuntimeError('Selected torch runtime does not expose CUDA')
    print('CUDA runtime:',torch.__version__,torch.cuda.get_device_name(0))
    sys.argv=[a.script]+a.args
    runpy.run_path(str(Path(a.script).resolve()),run_name='__main__')
if __name__=='__main__': main()
