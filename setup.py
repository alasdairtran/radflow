from setuptools import find_packages, setup

setup(name='radflow',
      version='0.1',
      description='Network of sequences',
      url='https://github.com/alasdairtran/radflow',
      author='Alasdair Tran',
      author_email='alasdair.tran@anu.edu.au',
      license='MIT',
      packages=find_packages(),
      install_requires=[],
      scripts=['bin/radflow'],
      zip_safe=False)
