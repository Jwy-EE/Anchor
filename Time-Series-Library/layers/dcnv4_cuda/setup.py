from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

def get_cuda_path():
    cuda_path = os.environ.get('CUDA_PATH', '')
    if cuda_path and os.path.exists(cuda_path):
        return cuda_path
    
    common_paths = [
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8',
        '/usr/local/cuda',
        '/usr/local/cuda-12.1',
        '/usr/local/cuda-11.7',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return ''

ext_modules = [
    CUDAExtension(
        name='dcnv4_1d_cuda_backend',
        sources=[
            'cuda/dcnv4_1d_ops.cpp',
            'cuda/dcnv4_1d_kernel.cu',
        ],
        include_dirs=[
            os.path.join(get_cuda_path(), 'include') if get_cuda_path() else '',
        ],
        library_dirs=[
            os.path.join(get_cuda_path(), 'lib/x64') if get_cuda_path() else '',
        ],
        libraries=['cudart'],
        extra_compile_args={
            'cxx': [
                '-O3',
                '-Wall',
                '-std=c++17',
                '-D_GLIBCXX_USE_CXX11_ABI=1',
            ],
            'nvcc': [
                '-O3',
                '-Xcompiler', '-Wall',
                '-std=c++17',
                '--use_fast_math',
                '--ptxas-options=-v',
                '--maxrregcount=64',
                '-gencode', 'arch=compute_61,code=sm_61',
                '-gencode', 'arch=compute_75,code=sm_75',
                '-gencode', 'arch=compute_86,code=sm_86',
                '-gencode', 'arch=compute_89,code=sm_89',
            ]
        }
    )
]

try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = 'DCNv4 1D CUDA扩展'

setup(
    name='dcnv4-1d-cuda',
    version='0.1.0',
    author='Time-Series-Library',
    author_email='',
    description='DCNv4 1D CUDA扩展',
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    
    install_requires=[
        'torch>=2.0.0',
        'ninja>=1.10.0',
    ],
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    python_requires='>=3.8',
    
    package_data={
        'dcnv4_cuda': [
            'cuda/*.cu',
            'cuda/*.cuh',
            'cuda/*.cpp',
        ],
    },
    
    scripts=[],
    
    project_urls={
        'Source': 'https://github.com/yourusername/time-series-library',
        'Bug Reports': 'https://github.com/yourusername/time-series-library/issues',
    },
)
