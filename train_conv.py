import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle

from mnist_data import *
from model_conv import ConvVAE

'''
vae implementation, alpha version, used with mnist

LOADS of help was taken from:

https://jmetzen.github.io/2015-11-27/vae.html

Ignore this file.
'''

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--training_epochs', type=int, default=300,
                     help='training epochs')
  parser.add_argument('--display_step', type=int, default=10000,
                     help='display step')
  parser.add_argument('--checkpoint_step', type=int, default=5,
                     help='checkpoint step')
  parser.add_argument('--batch_size', type=int, default=100,
                     help='batch size')
  parser.add_argument('--z_dim', type=int, default=2,
                     help='z dim')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                     help='learning rate')
  args = parser.parse_args()
  return train(args)

def train(args):

  learning_rate = args.learning_rate
  batch_size = args.batch_size
  training_epochs = args.training_epochs
  display_step = args.display_step
  checkpoint_step = args.checkpoint_step # save training results every check point step
  z_dim = args.z_dim # number of latent variables.

  dirname = 'save'
  if not os.path.exists(dirname):
    os.makedirs(dirname)

  with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
    cPickle.dump(args, f)

  vae = ConvVAE(learning_rate=learning_rate, batch_size=batch_size, z_dim = z_dim)

  mnist = read_data_sets()
  n_samples = mnist.num_examples

  # load previously trained model if appilcable
  ckpt = tf.train.get_checkpoint_state(dirname)
  if ckpt:
    vae.load_model(dirname)

  # Training cycle
  for epoch in range(training_epochs):
    avg_cost = 0.
    mnist.shuffle_data()
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
      batch_xs = mnist.next_batch(batch_size)

      # Fit training using batch data
      cost, l2_loss, kl_loss = vae.partial_fit(batch_xs)

      # Display logs per epoch step
      if i % display_step == 0:
        print "Epoch:", '%04d' % (epoch+1), \
              "batch:", '%04d' % (i), \
              "cost =", "{:.6f}".format(cost), \
              "l2_loss =", "{:.6f}".format(l2_loss), \
              "kl_loss =", "{:.6f}".format(kl_loss)

      # Compute average loss
      avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    print "Epoch:", '%04d' % (epoch+1), \
          "cost=", "{:.6f}".format(avg_cost)

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
