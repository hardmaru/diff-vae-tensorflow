import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle

from mnist_data import *
from model import VAE

'''
vae implementation, alpha version, used with mnist

LOADS of help was taken from:

https://jmetzen.github.io/2015-11-27/vae.html

'''

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--training_epochs', type=int, default=350,
                     help='training epochs')
  parser.add_argument('--checkpoint_step', type=int, default=5,
                     help='checkpoint step')
  parser.add_argument('--batch_size', type=int, default=500,
                     help='batch size')
  parser.add_argument('--z_dim', type=int, default=8,
                     help='z dim')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                     help='learning rate')
  parser.add_argument('--keep_prob', type=float, default=0.95,
                     help='dropout keep probability')
  parser.add_argument('--diff_mode', type=int, default=0,
                     help='experimental differentiation mode. 1 = true, 0 = false')
  parser.add_argument('--loss_mode', type=int, default=1,
                     help='1 = l2 loss, 2 = bournoulli loss')
  args = parser.parse_args()
  return train(args)

def train(args):

  learning_rate = args.learning_rate
  batch_size = args.batch_size
  training_epochs = args.training_epochs
  keep_prob = args.keep_prob
  checkpoint_step = args.checkpoint_step # save training results every check point step
  z_dim = args.z_dim # number of latent variables.
  loss_mode = args.loss_mode

  diff_mode = False
  if args.diff_mode == 1:
    diff_mode = True

  dirname = 'save'
  if not os.path.exists(dirname):
    os.makedirs(dirname)

  with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
    cPickle.dump(args, f)

  vae = VAE(learning_rate=learning_rate, batch_size=batch_size, z_dim = z_dim, keep_prob = keep_prob, loss_mode = loss_mode)

  mnist = read_data_sets()
  n_samples = mnist.num_examples

  # load previously trained model if appilcable
  ckpt = tf.train.get_checkpoint_state(dirname)
  if ckpt:
    vae.load_model(dirname)

  # Training cycle
  for epoch in range(training_epochs):
    avg_cost = 0.
    avg_likelihood_loss = 0.
    avg_kl_loss = 0.
    mnist.shuffle_data()
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
      batch_xs = mnist.next_batch(batch_size)

      if (diff_mode == True):
        batch_xs = mnist.integrate_batch(batch_xs)

      # Fit training using batch data
      cost, likelihood_loss, kl_loss = vae.partial_fit(batch_xs)

      # Compute average loss
      avg_cost += cost / n_samples * batch_size
      avg_likelihood_loss += likelihood_loss / n_samples * batch_size
      avg_likelihood_loss += kl_loss / n_samples * batch_size

      # Display logs per batch
      '''
      print "batch:", '%04d' % (i+1), \
            "total loss =", "{:.6f}".format(cost), \
            "likelihood_loss =", "{:.6f}".format(likelihood_loss), \
            "kl_loss =", "{:.6f}".format(kl_loss)
      '''

    # Display logs per epoch step
    print "Epoch:", '%04d' % (epoch+1), \
          "total loss =", "{:.6f}".format(avg_cost), \
          "likelihood_loss =", "{:.6f}".format(avg_likelihood_loss), \
          "kl_loss =", "{:.6f}".format(avg_kl_loss)

    # save model
    if epoch > 0 and epoch % checkpoint_step == 0:
      checkpoint_path = os.path.join('save', 'model.ckpt')
      vae.save_model(checkpoint_path, epoch)
      print "model saved to {}".format(checkpoint_path)

  # save model one last time, under zero label to denote finish.
  vae.save_model(checkpoint_path, 0)

  return vae

if __name__ == '__main__':
  main()
