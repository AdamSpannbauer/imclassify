from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

version = {}
with open("vidstab/version.py") as f:
    exec(f.read(), version)

setup(name='imclassify',
      version=version['__version__'],
      description='Quick end-to-end framework for building an image classifier.',
      author='Adam Spannbauer',
      author_email='spannbaueradam@gmail.com',
      url='https://github.com/AdamSpannbauer/imclassify',
      packages=['imclassify'],
      license='MIT',
      install_requires=[
          'keras',
          'tensorflow',
          'sklearn',
          'imutils',
          'numpy',
          'tqdm',
          'h5py',
      ],
      extras_require={
          'cv2': ['opencv-contrib-python >= 3.4.0']
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ]
      )
