# diff-vae-tensorflow

<a href="url"><img src="https://cdn.rawgit.com/hardmaru/diff-vae-tensorflow/master/img/1.png" align="left" width="320" ></a>

[Variational Autoencoder](https://arxiv.org/abs/1312.6114) implementation in tensorflow based off another [example](https://jmetzen.github.io/2015-11-27/vae.html).  In addition, `Sampler` class to interactively work with results inside IPython.

## How to use

Default parameters should be fine for MNIST.  Both L2 loss and logistic regression loss (Bernoulli) supported.  Default dataset is MNIST.

To train a model:

```
python train.py --help
```

To use the model, load IPython

```python
%run -i sample.py
sampler = Sampler() # loads trained model inside /save
```

To generate a random MNIST

```python
z = sampler.generate_z() # generates iid normal latent variable of 8 dimensions
m = sampler.generate(z) # generates a sample image from latent variables
sampler.show_image(m) # displays the image from the prompt
```
<a href="url"><img src="https://cdn.rawgit.com/hardmaru/diff-vae-tensorflow/master/img/a.png" align="left" width="320" ></a>

Alternatively, we can generate and display the image in one line:

```python
sampler.show_image_from_z(sampler.generate_z()) # displays the image from the prompt
```
<a href="url"><img src="https://cdn.rawgit.com/hardmaru/diff-vae-tensorflow/master/img/b.png" align="left" width="320" ></a>

We can draw a random image from the MNIST database, display it, and also display the autoencoded reconstruction:

```python
m = sampler.get_random_mnist() # get a random real MNIST image
sampler.show_image(m) # display the image
z = sampler.encode(m) # encode m into latent variables z
sampler.show_image_from_z(z) # show the autoencoded image
```
<a href="url"><img src="https://cdn.rawgit.com/hardmaru/diff-vae-tensorflow/master/img/0.png" align="left" width="320" ></a>

There are also some operations to perform image processing on images.  For example, if we want to differentiate the image, ie, find d(m)/dxdy:
```python
m = sampler.get_random_mnist() # get a random real MNIST image
diff_m = sampler.diff_image(m)
integrate_m = sampler.integrate_image(diff_m)
recover_m = sampler.integrate_image(diff_m)
sampler.show_image(m)
sampler.show_image(diff_m)
sampler.show_image(integrate_m)
recover_m = sampler.diff_image(integrate_m)
sampler.show_image(recover_m)
recover_m = sampler.integrate_image(diff_m)
sampler.show_image(recover_m) # same as previous image
```

<a href="url"><img src="https://cdn.rawgit.com/hardmaru/diff-vae-tensorflow/master/img/2.png" align="left" width="320" ></a>
<a href="url"><img src="https://cdn.rawgit.com/hardmaru/diff-vae-tensorflow/master/img/3.png" align="left" width="320" ></a>

# License

MIT - everything else
