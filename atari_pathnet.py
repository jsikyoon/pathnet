# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time
import sys

from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
import pathnet
import argparse

FLAGS=None;
log_dir=None;

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

def train():
  #initial learning rate
  initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                      INITIAL_ALPHA_HIGH,
                                      INITIAL_ALPHA_LOG_RATE)

  # parameter server and worker information
  ps_hosts = np.zeros(FLAGS.ps_hosts_num,dtype=object);
  worker_hosts = np.zeros(FLAGS.worker_hosts_num,dtype=object);
  port_num=FLAGS.st_port_num;
  for i in range(FLAGS.ps_hosts_num):
    ps_hosts[i]=str(FLAGS.hostname)+":"+str(port_num);
    port_num+=1;
  for i in range(FLAGS.worker_hosts_num):
    worker_hosts[i]=str(FLAGS.hostname)+":"+str(port_num);
    port_num+=1;
  ps_hosts=list(ps_hosts);
  worker_hosts=list(worker_hosts);
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  
  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
  
  
  if FLAGS.job_name == "ps":
    server.join();
  elif FLAGS.job_name == "worker":
    device=tf.train.replica_device_setter(
          worker_device="/job:worker/task:%d" % FLAGS.task_index,
          cluster=cluster);
    
    learning_rate_input = tf.placeholder("float")
    
    grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                  decay = RMSP_ALPHA,
                                  momentum = 0.0,
                                  epsilon = RMSP_EPSILON,
                                  clip_norm = GRAD_NORM_CLIP,
                                  device = device)
    
    tf.set_random_seed(1);
    #There are no global network
    training_thread = A3CTrainingThread(0, "", initial_learning_rate,
                                          learning_rate_input,
                                          grad_applier, MAX_TIME_STEP,
                                          device = device,FLAGS=FLAGS,task_index=FLAGS.task_index)
    
    # prepare session
    with tf.device(tf.train.replica_device_setter(
          worker_device="/job:worker/task:%d" % FLAGS.task_index,
          cluster=cluster)):
      flag = tf.get_variable('flag',[],initializer=tf.constant_initializer(0),trainable=False);
      flag_ph=tf.placeholder(flag.dtype,shape=flag.get_shape());
      flag_ops=flag.assign(flag_ph);
      global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False);
      global_step_ph=tf.placeholder(global_step.dtype,shape=global_step.get_shape());
      global_step_ops=global_step.assign(global_step_ph);
      score = tf.get_variable('score',[],initializer=tf.constant_initializer(-21),trainable=False);
      score_ph=tf.placeholder(score.dtype,shape=score.get_shape());
      score_ops=score.assign(score_ph);
      score_set=np.zeros(FLAGS.worker_hosts_num,dtype=object);
      score_set_ph=np.zeros(FLAGS.worker_hosts_num,dtype=object);
      score_set_ops=np.zeros(FLAGS.worker_hosts_num,dtype=object);
      for i in range(FLAGS.worker_hosts_num):
        score_set[i] = tf.get_variable('score'+str(i),[],initializer=tf.constant_initializer(-1000),trainable=False);
        score_set_ph[i]=tf.placeholder(score_set[i].dtype,shape=score_set[i].get_shape());
        score_set_ops[i]=score_set[i].assign(score_set_ph[i]);
      init_op=tf.global_variables_initializer();
      # summary for tensorboard
      tf.summary.scalar("score", score);
      summary_op = tf.summary.merge_all()
      saver = tf.train.Saver();
    
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                   global_step=global_step,
                                   logdir=FLAGS.log_dir,
                                   summary_op=summary_op,
                                   saver=saver,
                                   init_op=init_op)
    
    with sv.managed_session(server.target) as sess:
      if(FLAGS.task_index!=(FLAGS.worker_hosts_num-1)):
        while True:
          if(sess.run([flag])[0]==1):
            break;
            time.sleep(2);
        # set start_time
        wall_t=0.0;
        start_time = time.time() - wall_t
        training_thread.set_start_time(start_time)
        while True:
          if sess.run([global_step])[0] > MAX_TIME_STEP:
            break
          diff_global_t = training_thread.process(sess, sess.run([global_step])[0], "",
                                                summary_op, "",score_ph,score_ops,"",FLAGS,score_set_ph[FLAGS.task_index],score_set_ops[FLAGS.task_index])
          sess.run(global_step_ops,{global_step_ph:sess.run([global_step])[0]+diff_global_t});
          #print(str(FLAGS.task_index)+","+str(sess.run([global_step])[0]));
      else:
        # Generating randomly geopath
        geopath_set=np.zeros(FLAGS.worker_hosts_num-1,dtype=object);
        for i in range(FLAGS.worker_hosts_num-1):
          geopath_set[i]=pathnet.get_geopath(FLAGS.L,FLAGS.M,FLAGS.N);
          pathnet.geopath_insert(sess,training_thread.local_network.geopath_update_placeholders_set[i],training_thread.local_network.geopath_update_ops_set[i],geopath_set[i],FLAGS.L,FLAGS.M);
        print("Geopath Setting Done");
        sess.run(flag_ops,{flag_ph:1});

        score_subset=np.zeros(FLAGS.B,dtype=float);
        score_set_print=np.zeros(FLAGS.worker_hosts_num,dtype=float);
        rand_idx=range(FLAGS.worker_hosts_num-1); #np.random.shuffle(rand_idx);
        rand_idx=rand_idx[:FLAGS.B];
        while True:
          flag_sum=0;
          for i in range(FLAGS.worker_hosts_num-1):
            score_set_print[i]=sess.run([score_set[i]])[0];
          for i in range(len(rand_idx)):
            score_subset[i]=sess.run([score_set[rand_idx[i]]])[0];
            if(score_subset[i]==-1000):
              flag_sum=1;
              break;
          if(flag_sum==0):
            winner_idx=rand_idx[np.argmax(score_subset)];
            #print(str(sess.run([global_step])[0])+" Step Score: "+str(sess.run([score_set[winner_idx]])[0]));
            for i in rand_idx:
              if(i!=winner_idx):
                geopath_set[i]=np.copy(geopath_set[winner_idx]);
                geopath_set[i]=pathnet.mutation(geopath_set[i],FLAGS.L,FLAGS.M,FLAGS.N);
              sess.run(score_set_ops[i],{score_set_ph[i]:-1000});
            rand_idx=range(FLAGS.worker_hosts_num-1); #np.random.shuffle(rand_idx);
            rand_idx=rand_idx[:FLAGS.B];
          else:
            time.sleep(2);
    sv.stop();
    print("Done");

def main(_):
  FLAGS.log_dir+=str(int(time.time()));
  FLAGS.ps_hosts_num+=1;
  FLAGS.worker_hosts_num+=1;
  train()
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts_num",
      type=int,
      default=5,
      help="The Number of Parameter Servers"
  )
  parser.add_argument(
      "--worker_hosts_num",
      type=int,
      default=10,
      help="The Number of Workers"
  )
  parser.add_argument(
      "--hostname",
      type=str,
      default="seltera46",
      help="The Hostname of the machine"
  )
  parser.add_argument(
      "--st_port_num",
      type=int,
      default=2222,
      help="The start port number of ps and worker servers"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  parser.add_argument('--log_dir', type=str, default='/tmp/a3c_log/',
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
  parser.add_argument('--B', type=int, default=3,
                      help='The Number of Candidates for each competition')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
