from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='arp',
      packages=[package for package in find_packages()
                if package.startswith('arp')],
      install_requires=[
          'numpy',
          'matplotlib'
      ],
      description='Autoregressive Policies',
      author='Kindred AI',
      url='https://github.com/dkorenkevych/arp',
      version='1.0')