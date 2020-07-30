
from setuptools import setup

setup(
      name='sinn',
      version='0.2.0dev',
      description="A package for Simulation and Inference of Neuron Networks",

      author="Alexandre René",
      author_email="arene010@uottawa.ca",

      license='MIT',

      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering :: Information Analysis'
      ],

      packages=['sinn'],

      install_requires=['numpy >= 1.13',
                        'scipy',
                        'pydantic',
                        'mackelab_toolbox >= 0.1.0dev2',
                        'theano_shim >= 0.2.3',
                        'theano',
                        'tqdm',
                        'packaging',
                        'odictliteral'],
      # Add theano as optional dependency

      extras_require = {
        'dev': [
            'pytest',
            'pytest-xdist'
        ]
      }
     )
