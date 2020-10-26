from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

print(find_packages())

setup(name='vschaos',
      version='0.1.0',
      description='lavish audio synthesis from unfathomable variational abysses',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/domkkirke/vschaos_package',
      author='domkirke',
      author_email='domkirke@wolfgang.wang',
      license='GPLv3',
      packages=find_packages(),
      install_requires=['numpy',
        'torch',
        'torchvision',
        'torchaudio',
        'torch-dct',
        'librosa',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'dill',
        'ffmpeg',
        'scipy',
        'scikit-image',
        'python-osc',
        'tensorboard'
     ],
     classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Development Status :: 3 - Alpha',
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent"
     ],
     python_requires='>=3.6',
     keywords='machine learning, audio synthesis, generative models, variational, interactive',
     zip_safe=False)
