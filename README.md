# vschaos_package

The vschaos library is a high-level library for audio variational synthesis based on torch. For further information and
audio examples, please see [the corresponding support page.](https://domkirke.github.io/vschaos_package/)
This is the package version of the vschaos library, that you can find [at the following address](https://github.com/domkirke/vschaos).

# Installation
To install vschaos, please enter the following instructions into your terminal : 
````bash
$ git clone --recursive https://github.com/domkirke/vschaos_package.git 
$ cd vschaos_package
$ python3 setup.py install
````

**Important** : the `setup.py` file installs secondary dependencies for the library, but you will have to install the librairies `pytorch` and `tensorflow` manually. This way, you will be able to install these packages by choosing yourself the CUDA setup in case.  

It is strongly advised to install `vschaos_package` in a specific python environment. The pacakge has been tested using the [microconda](https://docs.conda.io/en/latest/miniconda.html) environment manager.

## Requirements for Max playgrounds

For using the playgrounds, you will have to install the followings librairies : 
* Install the latest [bach and dada](https://www.bachproject.net/dl/) libraries 
* Install the [MuBu](http://forumnet.ircam.fr/shop/fr/forumnet/59-mubu-pour-max.html) library for Max 
* Install the [CNMAT externals](https://cnmat.berkeley.edu/downloads) for Max

You can set up the target python3 executable by editing the file `max/plugin/python_env.txt`. 
Example : 
```
1, /Users/**/anaconda3/bin/python3;
```
Do not forget to add the folder `vschaos/max` to your File Preferences in Max. Note that the `Sine Bank` STFT inversion option is only available in Max 8.   


# Getting Started 


## Datasets and pretrained models

Some datasets and pre-trained models are available [here](https://www.dropbox.com/sh/cmb9jjd218fcwj6/AADCtHyeBSBoSZqEtwMrNokZa?dl=0). 

## Tutorials
To get started, you can follow the tutorials files and notebooks present in [docs/tutorials](https://github.com/domkirke/vschaos_package/tree/master/docs/tutorials). A full reference will be released with the first official release of `vschaos` ; stay tuned!

## Command-line training
The file `spectral_train.py` can be used to easily train custom models, giving access to most functionalities of `vschaos` through command-line arguments. **However**, you should first read the [first tutorial](https://github.com/domkirke/vschaos_package/blob/release/docs/tutorials/1_datasets.py) to know how to wrap your data for being accurately loaded by the library. For a complete view of available options, you can type in your terminal
```bash
$ python spectral_train.py -h
```
As the number of available options can be a little overwhelmed, here are some examples for some basic models.

* **Simple AE**
```bash
$ python spectral_train.py --dbroot *your_data* --transform *your_transform* --dims 16 --hidden_dims 800 --hidden_num 2 --epochs 300 --lr 1e-4 --beta 0 --save_epochs 40 --plot_epochs 20 
```
will train a simple auto-encoder with 2-layers MLPs with 800 hidden units using a learning rate of 1e-4 during 300 epochs. Regularization set is canceled by setting beta to 0. Model will be saved every 40 epochs, and monitored every 20 epochs. 

* **VAE with MLP**
```bash
$ python spectral_train.py --dbroot *your_data* --transform *your_transform* --dims 16 --hidden_dims 800 --hidden_num 2 --epochs 300 --lr 1e-4 --save_epochs 40 --plot_epochs 20 --warmup 50
```
will train a VAE with 2-layers MLPs with 800 hidden units using a learning rate of 1e-4 during 300 epochs and a warmup of KLD for 50 epochs. 

* **VAE with convolutional module**

```bash
$ python spectral_train.py --dbroot *your_data* --transform *your_transform* --dims 16 --hidden_dims 600 --hidden_num 1 --channels 32 32 32 --kernel_size 15 11 7 --stride 1 2 2 --epochs 300 --lr 1e-4 --save_epochs 40 --plot_epochs 20
```
will train a VAE with 3 convolutional layers of 32 channels and decreasing kernel sizes from 15 to 7 (and increasing strides for downsampling), as well as a flattening MLP of 600 units. 

* **Conditioned VAE**
```bash
$ python spectral_train.py --dbroot *additive_data* --tasks f0 n_partials harmonic_decay --transform *your_transform* --dims 16 --hidden_dims 800 --hidden_num 2 --conditioning f0 --conditioning_target decoder --epochs 300 --lr 1e-4 --save_epochs 40 --plot_epochs 20
```

will train a VAE with 2-layers MLPs conditioned on the `f0` metadata of the `toy_additive_mini`dataset. Plots will be done with all the indicated tasks. 

* **VAE with normalizing flows**
```bash
$ python spectral_train.py --dbroot *additive_data* --tasks f0 n_partials harmonic_decay --transform *your_transform* --dims 16 --hidden_dims 800 --hidden_num 2 --flow_blocks PlanarFlow --flow_length 10 --epochs 300 --lr 1e-4 --save_epochs 40 --plot_epochs 20
```

* **Conditioined multi-layer VAE**
```bash
$ python spectral_train.py --dbroot *additive_data* --tasks f0 n_partials harmonic_decay --transform *your_transform* --dims 16 8 --hidden_dims 800 300 --hidden_num 2 2 --conditioning f0 --conditioning_layer 1 --conditioning_target decoder --flow_blocks PlanarFlow --flow_length 10 --epochs 300 --lr 1e-4 --save_epochs 40 --plot_epochs 20
```

will train a 2-layered VAE with 2-layers MLPs, with only the decoder between latent layers being conditioned on the `f0` task. 

## Some tips

* when using the script `spectral_train.py`, you can add the option `--files 100` to only load by example 100 files of your dataset, and add the `--check 1` option to make a pause before training the model. This is useful to check quickly if the training is really the one you wanted, and hence prevent to waste your time!

* by default, the non-linearity at the output is `softplus`, as most of the models are trained on magnitude spectra. If you try to learning something else, please remind this!
 
