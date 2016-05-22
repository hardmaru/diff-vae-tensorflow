'''

Sampler Class

This file is meant to be run inside an IPython session, as it is meant
to be used interacively for experimentation.

It shouldn't be that hard to take bits of this code into a normal
command line environment though if you want to use outside of IPython.

usage:

%run -i sampler.py

sampler = Sampler()

'''

import numpy as np
import tensorflow as tf
import math
import random
import PIL
from PIL import Image
import pylab
from model import VAE
import matplotlib.pyplot as plt

mgc = get_ipython().magic
mgc(u'matplotlib inline')
mgc(u'run -i mnist_data.py')
pylab.rcParams['figure.figsize'] = (6.0, 6.0)

class Sampler():
  def __init__(self):
    self.mnist = None
    self.model = VAE()
    self.model.load_model('save')
    self.z = self.generate_z()
  def get_random_mnist(self, with_label = False):
    if self.mnist == None:
      self.mnist = read_data_sets()
    if with_label == True:
      data, label = self.mnist.next_batch(1, with_label)
      return data[0], label[0]
    return self.mnist.next_batch(1)[0]
  def get_random_specific_mnist(self, label = 2):
    m, l = self.get_random_mnist(with_label = True)
    for i in range(100):
      if l == label:
        break
      m, l = self.get_random_mnist(with_label = True)
    return m
  def generate_random_label(self, label):
    m = self.get_random_specific_mnist(label)
    self.show_image(m)
    self.show_image_from_z(self.encode(m))
  def generate_z(self):
    z = np.random.normal(size=(1, self.model.z_dim)).astype(np.float32)
    return z
  def encode(self, mnist_data):
    new_shape = [1]+list(mnist_data.shape)
    return self.model.transform(np.reshape(mnist_data, new_shape))
  def generate(self, z=None):
    if z is None:
      z = self.generate_z()
    else:
      z = np.reshape(z, (1, self.model.z_dim))
    self.z = z
    return self.model.generate(z)[0]
  def show_image(self, image_data):
    '''
    image_data is a tensor, in [height width depth]
    image_data is NOT the PIL.Image class
    '''
    plt.subplot(1, 1, 1)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = 1
    if c_dim > 1:
      plt.imshow(image_data, interpolation='nearest')
    else:
      plt.imshow(image_data.reshape(y_dim, x_dim), cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.show()
  def show_image_from_z(self, z):
    self.show_image(self.generate(z))
  def to_image(self, image_data):
    # convert to PIL.Image format from np array (0, 1)
    img_data = np.array(1-image_data)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = 1
    if c_dim > 1:
      img_data = np.array(img_data.reshape((y_dim, x_dim, c_dim))*255.0, dtype=np.uint8)
    else:
      img_data = np.array(img_data.reshape((y_dim, x_dim))*255.0, dtype=np.uint8)
    im = Image.fromarray(img_data)
    return im
  def diff_image(self, image_data):
    # perform 2d differentiation on mnist image
    m2 = np.array(image_data) # makes a copy
    m2[1:,1:,:] = m2[1:,1:,:]-m2[0:-1,1:,:]
    m2[1:,1:,:] = m2[1:,1:,:]-m2[1:,0:-1,:]
    return m2
  def integrate_image(self, image_data):
    # integrates differentiated batch back to mnist image
    m3 = np.array(image_data)
    m3 = m3.cumsum(axis=0)
    m3 = m3.cumsum(axis=1)
    return m3
