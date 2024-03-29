
from setuptools import setup, find_packages

setup(
      name='sinn',
      version='0.2.0rc3',
      description="A package for Simulation and Inference of Neuron Networks",

      author="Alexandre René",
      author_email="arene010@uottawa.ca",

      license='MIT',

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Topic :: Scientific/Engineering :: Information Analysis'
      ],

      packages=find_packages(),

      install_requires=['numpy >= 1.13',
                        'scipy',
                        'pydantic >= 1.7.1',
                        'parameters',
                        'mackelab-toolbox[iotools,utils,typing,parameters,theano] >= 0.2.0a1',
                        'tabulate',
                        'theano_shim >= 0.2.3',
                        'theano-pymc',
                        'tqdm',
                        'packaging',
                        'odictliteral'],
      # Add theano as optional dependency

      extras_require = {
        'dev': [
            'ipykernel',
            'pandas',
            'pint',
            'pytest',
            'pytest-xdist',
            'quantities',
            'matplotlib',
            'scipy',
            'seaborn',
            'sphinx',
            'sphinxcontrib.mermaid',
            'xarray'
        ]
      }
     )
