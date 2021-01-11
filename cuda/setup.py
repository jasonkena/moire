from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qrsolver",
    ext_modules=[
        CUDAExtension(
            "qrsolver",
            ["cusparseqr.cpp", "cusparseqr_kernel.cu",],
            libraries=["cusolver"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
