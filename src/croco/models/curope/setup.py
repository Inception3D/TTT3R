# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import sys
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

extra_link_args = []
if sys.platform == "win32" and IS_ROCM:
    # On Windows, MSVC-compiled objects that include <torch/extension.h>
    # generate a dllimport reference to c10::ValueError(SourceLocation, string)
    # via the inherited "using Error::Error" constructor.  c10.dll (built with
    # clang) does not export this decorated name.  Redirect it to the base
    # c10::Error(SourceLocation, string) constructor which IS exported.
    _val_imp = (
        "__imp_??0ValueError@c10@@QEAA@USourceLocation@1@"
        "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )
    _err_imp = (
        "__imp_??0Error@c10@@QEAA@USourceLocation@1@"
        "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )
    extra_link_args.append(f"/ALTERNATENAME:{_val_imp}={_err_imp}")

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
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
