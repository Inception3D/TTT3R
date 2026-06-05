# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
import torch

IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None

if IS_ROCM:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    extra_compile_args = dict(
        nvcc=["-O3"],
        cxx=["-O3"],
    )
else:
    from torch import cuda
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    # compile for all possible CUDA architectures
    all_cuda_archs = cuda.get_gencode_flags().replace("compute=", "arch=").split()
    # alternatively, you can list cuda archs that you want, eg:
    # all_cuda_archs = [
    # '-gencode', 'arch=compute_70,code=sm_70',
    # '-gencode', 'arch=compute_75,code=sm_75',
    # '-gencode', 'arch=compute_80,code=sm_80',
    # '-gencode', 'arch=compute_86,code=sm_86'
    # ]
    extra_compile_args = dict(
        nvcc=["-O3", "--ptxas-options=-v", "--use_fast_math"] + all_cuda_archs,
        cxx=["-O3"],
    )

setup(
    name="curope",
    ext_modules=[
        CUDAExtension(
            name="curope",
            sources=[
                "curope.cpp",
                "kernels.cu",
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
