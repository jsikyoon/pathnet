from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import pathnet
FLAGS = None

# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time

from game_ac_network import GameACPathNetNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU

global_t = 0
stop_requested = False
thread_rewards=0;
geopath_set=0;
summary_writer=None;
fixed_path=None;

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

def train():
  global global_t,thread_rewards, geopath_set, summary_writer, fixed_path;
  thread_rewards=np.zeros(FLAGS.candi,dtype=float);

  # device selection
  device = "/cpu:0"
  if USE_GPU:
    device = "/gpu:0"

  # learning rate selection
  initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                      INITIAL_ALPHA_HIGH,
                                      INITIAL_ALPHA_LOG_RATE)
 
  # prepare session
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                          allow_soft_placement=True))
  # global pathnet network 
  global_network = GameACPathNetNetwork(ACTION_SIZE, -1, device,FLAGS)
  # training thread 
  training_threads = []
  learning_rate_input = tf.placeholder("float")
  grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                decay = RMSP_ALPHA,
                                momentum = 0.0,
                                epsilon = RMSP_EPSILON,
                                clip_norm = GRAD_NORM_CLIP,
                                device = device)
  for i in range(FLAGS.candi):
    training_thread = A3CTrainingThread(i, global_network, initial_learning_rate,
                                        learning_rate_input,
                                        grad_applier, MAX_TIME_STEP,
                                        device = device,FLAGS=FLAGS);
    training_threads.append(training_thread)
  
  # initialization
  init = tf.global_variables_initializer()
  sess.run(init)
 
  # summary for tensorboard
  score_input = tf.placeholder(tf.int32)
  tf.summary.scalar("score", score_input)
  summary_op = tf.summary.merge_all()

  # Generating randomly geopath
  geopath_set=np.zeros(FLAGS.candi,dtype=object);
  for i in range(FLAGS.candi):
    geopath_set[i]=pathnet.get_geopath(FLAGS.L,FLAGS.M,FLAGS.N);

  # set wall time
  wall_t = 0.0

  def pathnet_train_function(parallel_index):
    global global_t,thread_rewards, fixed_path;
    training_thread = training_threads[parallel_index]
    # set start_time
    start_time = time.time() - wall_t
    training_thread.set_start_time(start_time)
    for task in range(2):
      while True:
        if stop_requested:
          break;
        if global_t > MAX_TIME_STEP:
          break;
        diff_global_t,reward,var_list = training_thread.process(sess, global_t, summary_writer,
                                              summary_op, score_input,geopath_set[parallel_index],FLAGS)
        global_t += diff_global_t;
        if(reward!=-1000):
          if(reward>thread_rewards[parallel_index]):
            thread_rewards[parallel_index]=reward;
      while True:
        if(global_t==0):
          break;
        time.sleep(2);
      training_thread.set_fixed_path(fixed_path);
     
  def pathnet_parameter_server(parallel_index):
    global geopath_set,thread_rewards,global_t, summary_writer, fixed_path;
    for task in range(2):
      print("=====Task"+str(task+1)+"==========");
      # Task summary
      summary_writer = tf.summary.FileWriter(FLAGS.log_dir+"task"+str(task+1), sess.graph);
      global_t=0;winner_idx=0;
      for i in range(FLAGS.candi):
        thread_rewards[i]=-1000;
      # random sampling of B candidates
      rand_idx=range(FLAGS.candi);np.random.shuffle(rand_idx);rand_idx=rand_idx[:FLAGS.B];
      while True:
        if stop_requested:
          break;
        if global_t > MAX_TIME_STEP:
          break;
        flag_sum=0;
        for i in rand_idx:
          if(thread_rewards[i]==-1000):
            flag_sum+=1;
        # After running of B candidates, copying geopath of winner and mutating the geopath of losers
        if(flag_sum==0):
          winner_idx=rand_idx[np.argmax(thread_rewards[rand_idx])];
          for i in rand_idx:
            if(i!=winner_idx):
              geopath_set[i]=np.copy(geopath_set[winner_idx]);
              geopath_set[i]=pathnet.mutation(geopath_set[i],FLAGS.L,FLAGS.M,FLAGS.N);
              thread_rewards[i]=-1000
          # random sampling of B candidates
          rand_idx=range(FLAGS.candi);np.random.shuffle(rand_idx);rand_idx=rand_idx[:FLAGS.B];
        time.sleep(5);
      fixed_path=geopath_set[winner_idx];
      # fixed_path Setting
      for i in range(FLAGS.L):
        for j in range(FLAGS.M):
          if(fixed_path[i,j]==1.0):
            global_network.fixed_path[i,j]="1";
      #initialization
      tf.variables_initializer(global_network.get_vars());
      # Generating randomly geopath
      geopath_set=np.zeros(FLAGS.candi,dtype=object);
      for i in range(FLAGS.candi):
        geopath_set[i]=pathnet.get_geopath(FLAGS.L,FLAGS.M,FLAGS.N);
      
      
  def signal_handler(signal, frame):
    global stop_requested
    print('You pressed Ctrl+C!')
    stop_requested = True

  # Threading List    
  train_threads = []
  for i in range(FLAGS.candi):
    train_threads.append(threading.Thread(target=pathnet_train_function, args=(i,)))
  train_threads.append(threading.Thread(target=pathnet_parameter_server, args=(FLAGS.candi,)))
  
  signal.signal(signal.SIGINT, signal_handler)

  # set start time
  start_time = time.time() - wall_t
  
  ## First Task
  for t in train_threads:
    t.start()

  print('Press Ctrl+C to stop')
  signal.pause()

def main(_):
  FLAGS.log_dir+=str(int(time.time()))+"/";
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', type=str, default='/tmp/pathnet/atari/',
                      help='Summaries log directry')
  parser.add_argument('--M', type=int, default=10,
                      help='The Number of Modules per Layer')
  parser.add_argument('--L', type=int, default=4,
                      help='The Number of Layers')
  parser.add_argument('--N', type=int, default=4,
                      help='The Number of Selected Modules per Layer')
  parser.add_argument('--kernel_num', type=str, default='8,4,3',
                      help='The Number of Kernels for each layer')
  parser.add_argument('--stride_size', type=str, default='4,2,1',
                      help='Stride size for each layer')
  parser.add_argument('--candi', type=int, default=10,
                      help='The Number of Candidates of geopath')
  parser.add_argument('--B', type=int, default=3,
                      help='The Number of Candidates for each competition')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
