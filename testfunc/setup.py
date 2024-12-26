from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension
import torch

path_0 = torch.utils.cpp_extension.include_paths()
path_1 = '/opt/pytorch/pytorch/aten/src/ATen/core'
path_2 = '/opt/pytorch/pytorch/aten/src/ATen'
path_3 = '/opt/pytorch/pytorch/c10'
path_4 = '/opt/conda/envs/distserve/lib/python3.10/site-packages/torch/include'
path_5 = '/opt/pytorch/pytorch/torch'

lib_path_0 = torch.utils.cpp_extension.library_paths()
lib_path_1 = '/opt/conda/envs/distserve/lib'
lib_path_2 = '/opt/conda/envs/distserve/lib/python3.10/site-packages/torch/lib'
lib_path_3 = '/opt/conda/envs/distserve/lib/python3.10/site-packages'
lib_path_4 = '/opt/conda/envs/distserve/lib/python3.10/site-packages/torch'

ext_modules = [
    Extension(
        name='testfunc',
        sources=['testfunc.cpp','testfunc.wrapper.cpp','numa.cpp','CPUAllocator.cpp','Allocator.cpp','alloc_cpu.cpp'],
        include_dirs=[
            *path_0,
            path_1,
            path_2,
            path_3,
            path_4,
            path_5,
        ],
        library_dirs=[
            *lib_path_0,
            lib_path_1,
            lib_path_2,
            lib_path_3,
            lib_path_4,
        ],
        libraries=['numa','torch','python3.10','torch_python'],
    ),
]

setup(
    name='testfunc',
    version='0.1',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
