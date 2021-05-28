from setuptools import setup

setup(name='lsld2',
      version='0.0.6',
      description='TODO',
      long_description="Preliminary package structure for a simulation "
      "fitting tool",
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          # 'License :: OSI Approved :: MIT License', # TODO: License
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Chemistry',
      ],
      keywords='',
      url='https://github.com/Xiaoy01/lsld2',
      author='Sean Yang, Pranav Gupta',
      author_email='acert@cornell.edu',
      license='MIT',
      packages=['lsld2'],
      install_requires=[
          'markdown',
      ],
      include_package_data=True,
      zip_safe=False)
