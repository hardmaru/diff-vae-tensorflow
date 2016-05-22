import os
import time
from glob import glob
import tensorflow as tf

from ops import *

'''
conv vae implementation, alpha version, used with mnist, but easy to change to work with cifar-10

mnist samples cropped to 26x26 from 28x28 to produce more diverse training examples

LOADS of code sniplets was taken from:

https://github.com/carpedm20/DCGAN-tensorflow
https://jmetzen.github.io/2015-11-27/vae.html

Ignore this file.
'''

class ConvVAE():
  def __init__(self, batch_size=100,
                z_dim=2, gf_dim = 32, df_dim = 32,
                learning_rate = 0.01):
    """

    Args:
        sess: TensorFlow session
        batch_size: The size of batch. Should be specified before training.
        z_dim: (optional) Dimension of dim for Z. [20]
        gf_dim: (optional) Dimension of gen filters in first conv layer. [32]
        df_dim: (optional) Dimension of discrim filters in first conv layer. [32]
    """

    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.z_dim = z_dim
    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.x_dim = 26
    self.y_dim = 26
    self.n_points = self.x_dim * self.y_dim

    # tf Graph batch of image (batch_size, height, width, depth)
    self.x_raw = tf.placeholder(tf.float32, [batch_size, self.y_dim, self.x_dim, 1])

    # distort raw data (decided in the end to leave this task to DataLoader class)
    self.x = self.x_raw

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(batch_size, name='d_bn1')
    self.d_bn2 = batch_norm(batch_size, name='d_bn2')

    self.g_bn0 = batch_norm(batch_size, name='g_bn0')
    self.g_bn1 = batch_norm(batch_size, name='g_bn1')
    self.g_bn2 = batch_norm(batch_size, name='g_bn2')

    # Create autoencoder network
    self._create_network()
    # Define loss function based variational upper-bound and
    # corresponding optimizer
    self._create_loss_optimizer()

    # Initializing the tensor flow variables
    init = tf.initialize_all_variables()

    # Launch the session
    self.sess = tf.InteractiveSession()
    self.sess.run(init)
    self.saver = tf.train.Saver(tf.all_variables())

  def _create_network(self):


    # Use recognition network to determine mean and
    # (log) variance of Gaussian distribution in latent
    # space
    self.z_mean, self.z_log_sigma_sq = self._recognition_network(self.x)

    # Draw one sample z from Gaussian distribution
    n_z = self.z_dim
    eps = tf.random_normal((self.batch_size, n_z), 0.0, 1.0, dtype=tf.float32)
    # z = mu + sigma*epsilon
    self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

    # Use generator to determine mean of
    # Bernoulli distribution of reconstructed input
    self.x_reconstr_mean = self._generator_network(self.z)

  def _recognition_network(self, image):
    # Generate probabilistic encoder (recognition network), which
    # maps inputs onto a normal distribution in latent space.
    # The transformation is parametrized and can be learned.

    h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
    h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
    h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
    z_mean = linear(tf.reshape(h2, [self.batch_size, -1]), self.z_dim, 'd_h2_lin_mean')
    z_log_sigma_sq = linear(tf.reshape(h2, [self.batch_size, -1]), self.z_dim, 'd_h2_lin_log_sigma_sq')

    return (z_mean, z_log_sigma_sq)

  def _generator_network(self, z):
    # Generate probabilistic decoder (decoder network), which
    # maps points in latent space onto a Bernoulli distribution in data space.
    # The transformation is parametrized and can be learned.

    # project `z` and reshape
    self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*4*4*4, 'g_h0_lin', with_w=True)

    self.h0 = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 4])
    h0 = tf.nn.relu(self.g_bn0(self.h0))

    self.h1, self.h1_w, self.h1_b = deconv2d(h0,
        [self.batch_size, 7, 7, self.gf_dim*2], name='g_h1', with_w=True)
    h1 = tf.nn.relu(self.g_bn1(self.h1))

    h2, self.h2_w, self.h2_b = deconv2d(h1,
        [self.batch_size, 13, 13, self.gf_dim*1], name='g_h2', with_w=True)
    h2 = tf.nn.relu(self.g_bn2(h2))

    h3, self.h3_w, self.h3_b = deconv2d(h2,
        [self.batch_size, self.y_dim, self.x_dim, 1], name='g_h3', with_w=True)

    x_reconstr_mean = tf.nn.sigmoid(h3)
    return x_reconstr_mean

  def _create_loss_optimizer(self):
    # The loss is composed of two terms:
    # 1.) The reconstruction loss (the negative log probability
    #     of the input under the reconstructed Bernoulli distribution
    #     induced by the decoder in the data space).
    #     This can be interpreted as the number of "nats" required
    #     for reconstructing the input when the activation in latent
    #     is given.
    # Adding 1e-10 to avoid evaluatio of log(0.0)

    orig_image = tf.reshape(self.x, [self.batch_size, -1])
    new_image = tf.reshape(self.x_reconstr_mean, [self.batch_size, -1])

    '''
    reconstr_loss = \
        -tf.reduce_sum(orig_image * tf.log(1e-10 + new_image)
                       + (1-orig_image) * tf.log(1e-10 + 1 - new_image), 1)
    '''
    # use L2 loss instead:
    d = (orig_image - new_image)
    d2 = tf.mul(d, d) * 2.0
    self.vae_loss_l2 = tf.reduce_sum(d2, 1)

    self.vae_loss_l2 = tf.reduce_mean(self.vae_loss_l2) / self.n_points # average over batch and pixel

    # 2.) The latent loss, which is defined as the Kullback Leibler divergence
    ##    between the distribution in latent space induced by the encoder on
    #     the data and some prior. This acts as a kind of regularizer.
    #     This can be interpreted as the number of "nats" required
    #     for transmitting the the latent space distribution given
    #     the prior.
    self.vae_loss_kl = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                       - tf.square(self.z_mean)
                                       - tf.exp(self.z_log_sigma_sq), 1)

    self.vae_loss_kl = tf.reduce_mean(self.vae_loss_kl) / self.n_points

    self.cost = self.vae_loss_l2 + self.vae_loss_kl   # average over batch

    #self.cost = tf.reduce_mean(self.vae_loss_kl + self.vae_loss_l2)

    self.t_vars = tf.trainable_variables()

    # Use ADAM optimizer
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost, var_list=self.t_vars)

  def partial_fit(self, X):
    """Train model based on mini-batch of input data.

    Return cost of mini-batch.
    """

    opt, cost, vae_loss_l2, vae_loss_kl = self.sess.run((self.optimizer, self.cost, self.vae_loss_l2, self.vae_loss_kl),
                              feed_dict={self.x_raw: X})
    return cost, vae_loss_l2, vae_loss_kl

  def transform(self, X):
    """Transform data by mapping it into the latent space."""
    # Note: This maps to mean of distribution, we could alternatively
    # sample from Gaussian distribution
    return self.sess.run(self.z_mean, feed_dict={self.x_raw: X})

  def generate(self, z_mu=None):
    """ Generate data by sampling from latent space.

    If z_mu is not None, data for this point in latent space is
    generated. Otherwise, z_mu is drawn from prior in latent
    space.
    """
    if z_mu is None:
        z_mu = np.random.normal(size=(1, self.z_dim))
    # Note: This maps to mean of distribution, we could alternatively
    # sample from Gaussian distribution
    return self.sess.run(self.x_reconstr_mean,
                           feed_dict={self.z: z_mu})

  def reconstruct(self, X):
    """ Use VAE to reconstruct given data. """
    return self.sess.run(self.x_reconstr_mean,
                         feed_dict={self.x_raw: X})

  def save_model(self, checkpoint_path, epoch):
    """ saves the model to a file """
    self.saver.save(self.sess, checkpoint_path, global_step = epoch)

  def load_model(self, checkpoint_path):

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print "loading model: ",ckpt.model_checkpoint_path

    self.saver.restore(self.sess, ckpt.model_checkpoint_path)

