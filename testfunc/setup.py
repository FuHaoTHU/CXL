from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch
import pybind11

path_0 = torch.utils.cpp_extension.include_paths()
path_1 = '/opt/conda/envs/distserve/lib/python3.10/site-packages/torch/include/c10/core'
path_2 = '/opt/conda/envs/distserve/lib/python3.10/site-packages/torch/include/c10/utils'
path_3 = '/opt/conda/envs/distserve/lib/python3.10/site-packages/torch/include/torch/csrc'
path_4 = '/opt/conda/envs/distserve/lib/python3.10/site-packages/torch/include/ATen'

lib_path_0 = torch.utils.cpp_extension.library_paths()
lib_path_1 = '/opt/conda/envs/distserve/lib'
lib_path_2 = '/opt/conda/envs/distserve/lib/python3.10/site-packages/torch/lib'
lib_path_3 = '/opt/conda/envs/distserve/lib/python3.10/site-packages'
lib_path_4 = '/opt/conda/envs/distserve/lib/python3.10/site-packages/torch'

ext_modules = [
    CppExtension(
        name='testfunc',
        sources=['testfunc.cpp','testfunc.wrapper.cpp','numa.cpp','CPUAllocator.cpp','Allocator.cpp','alloc_cpu.cpp'],
        include_dirs=[
            pybind11.get_include(),
            *path_0,
            path_1,
            path_2,
            path_3,
            path_4,
        ],
        library_dirs=[
            *lib_path_0,
            lib_path_1,
            lib_path_2,
            lib_path_3,
        ],
        libraries=['numa','torch_python','c10','torch','python3.10','torch_cpu','torch_cuda'],
        language='c++',
    ),
]

setup(
    name='testfunc',
    version='0.1',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
