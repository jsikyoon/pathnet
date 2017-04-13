from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import input_data
import pathnet

import numpy as np

FLAGS = None


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)
  total_tr_data, total_tr_label = mnist.train.next_batch(mnist.train._num_examples);
  
  # Gathering a1 Data
  tr_data_a1=total_tr_data[(total_tr_label[:,FLAGS.a1]==1.0)];
  for i in range(len(tr_data_a1)):
    for j in range(len(tr_data_a1[0])):
      rand_num=np.random.rand();
      if(rand_num>=0.5):
        tr_data_a1[i,j]=np.minimum(tr_data_a1[i,j]+rand_num,1.0);
  
  # Gathering a2 Data
  tr_data_a2=total_tr_data[(total_tr_label[:,FLAGS.a2]==1.0)];
  for i in range(len(tr_data_a2)):
    for j in range(len(tr_data_a2[0])):
      rand_num=np.random.rand();
      if(rand_num>=0.5):
        tr_data_a2[i,j]=np.minimum(tr_data_a2[i,j]+rand_num,1.0);
  
  # Gathering b1 Data
  tr_data_b1=total_tr_data[(total_tr_label[:,FLAGS.b1]==1.0)];
  for i in range(len(tr_data_b1)):
    for j in range(len(tr_data_b1[0])):
      rand_num=np.random.rand();
      if(rand_num>=0.5):
        tr_data_b1[i,j]=np.minimum(tr_data_b1[i,j]+rand_num,1.0);

  # Gathering b2 Data
  tr_data_b2=total_tr_data[(total_tr_label[:,FLAGS.b2]==1.0)];
  for i in range(len(tr_data_b2)):
    for j in range(len(tr_data_b2[0])):
      rand_num=np.random.rand();
      if(rand_num>=0.5):
        tr_data_b2[i,j]=np.minimum(tr_data_b2[i,j]+rand_num,1.0);
  
  ## TASK 1
  sess = tf.InteractiveSession()

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 2)

  # geopath_examples
  with tf.name_scope('geopath'):
    geopath= tf.placeholder(tf.float32, [FLAGS.L,FLAGS.M], name='geopath');
  
  # fixed weights list
  fixed_list=np.ones((FLAGS.L,FLAGS.M),dtype=str);
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      fixed_list[i,j]='0';    

  # Hidden Layers
  weights_list=np.zeros((FLAGS.L,FLAGS.M),dtype=object);
  biases_list=np.zeros((FLAGS.L,FLAGS.M),dtype=object);
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      if(i==0):
        weights_list[i,j]=pathnet.module_weight_variable([784,FLAGS.filt]);
        biases_list[i,j]=pathnet.module_bias_variable([FLAGS.filt]);
      else:
        weights_list[i,j]=pathnet.module_weight_variable([FLAGS.filt,FLAGS.filt]);
        biases_list[i,j]=pathnet.module_bias_variable([FLAGS.filt]);
 
  zeros=tf.constant(0.0,shape=[FLAGS.filt]); 
  for i in range(FLAGS.L):
    layer_modules_list=np.zeros(FLAGS.M,dtype=object);
    for j in range(FLAGS.M):
      if(i==0):
        layer_modules_list[j]=tf.cond(tf.equal(geopath[i,j],1.0),lambda: pathnet.module(x, weights_list[i,j], biases_list[i,j], 'layer'+str(i+1)+"_"+str(j+1)), lambda: zeros);
      else:
        layer_modules_list[j]=tf.cond(tf.equal(geopath[i,j],1.0),lambda: pathnet.module2(j+1,net, weights_list[i,j], biases_list[i,j], 'layer'+str(i+1)+"_"+str(j+1)), lambda: zeros);
    net=np.sum(layer_modules_list);
 
  # Output Layer
  output_weights=pathnet.module_weight_variable([FLAGS.filt,2]);
  output_biases=pathnet.module_bias_variable([2]);
  y = pathnet.nn_layer(net,output_weights,output_biases,'output_layer');

  # Cross Entropy
  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)
  
  # Need to learn variables
  var_list_to_learn=[]+output_weights+output_biases;
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      if (fixed_list[i,j]=='0'):
        var_list_to_learn+=weights_list[i,j]+biases_list[i,j];
 
  # GradientDescent 
  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy,var_list=var_list_to_learn);

  # Accuracy 
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train1', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test1')
  tf.global_variables_initializer().run()
    
  # Make a TensorFlow feed_dict: maps data onto Tensor placeholders.
  def feed_dict(train,geopaths,batch_num=0,tr_flag1=0,tr_flag2=0):
    if train or FLAGS.fake_data:
      x_1=tr_data_a1[tr_flag1:tr_flag1+batch_num,:]; x_2=tr_data_a2[tr_flag2:tr_flag2+batch_num,:];
      if(len(x_1)<batch_num):
        x_1=np.append(x_1,tr_data_a1[:(tr_flag1+batch_num)%len(tr_data_a1),:],axis=0);
      if(len(x_2)<batch_num):
        x_2=np.append(x_2,tr_data_a2[:(tr_flag2+batch_num)%len(tr_data_a2),:],axis=0);
      xs=np.append(x_1,x_2,axis=0);
      ys=np.zeros((batch_num*2,2),dtype=float);
      for i in range(len(ys)):
        if(i<batch_num):
          ys[i,0]=1.0;
        else:
          ys[i,1]=1.0;
    return {x: xs, y_: ys, geopath: geopaths}

  # Generating randomly geopath
  geopath_set=np.zeros(FLAGS.candi,dtype=object);
  for i in range(FLAGS.candi):
    geopath_set[i]=pathnet.get_geopath(FLAGS.L,FLAGS.M,FLAGS.N);
  
  # parameters placeholders and ops 
  var_update_ops=np.zeros(len(var_list_to_learn),dtype=object);
  var_update_placeholders=np.zeros(len(var_list_to_learn),dtype=object);
  for i in range(len(var_list_to_learn)):
    var_update_placeholders[i]=tf.placeholder(var_list_to_learn[i].dtype,shape=var_list_to_learn[i].get_shape());
    var_update_ops[i]=var_list_to_learn[i].assign(var_update_placeholders[i]);
 
  tr_flag1=0; tr_flag2=0; data_num1=len(tr_data_a1); data_num2=len(tr_data_a2); 
  for i in range(FLAGS.max_steps):
    # Select Two Candidate to Tournament 
    first,second=pathnet.select_two_candi(FLAGS.candi);
    
    # First Candidate
    #pathnet.geopath_insert(sess,geopath_update_placeholders,geopath_update_ops,geopath_set[first],FLAGS.L,FLAGS.M);
    acc_geo1_tr=0;
    #var_list_backup=pathnet.parameters_backup(var_list_to_learn);
    #tr_flag_bak=tr_flag;
    for j in range(FLAGS.T):
      _, acc_geo1_tmp = sess.run([train_step,accuracy], feed_dict=feed_dict(True,geopath_set[first],int(FLAGS.batch_num/2),int(tr_flag1),int(tr_flag2)));
      tr_flag1=(tr_flag1+FLAGS.batch_num/2)%data_num1;
      tr_flag2=(tr_flag2+FLAGS.batch_num/2)%data_num2;
      acc_geo1_tr+=acc_geo1_tmp;
    #var_list_task1=pathnet.parameters_backup(var_list_to_learn);
    #tr_flag=tr_flag_bak;
    # Second Candidate
    #pathnet.geopath_insert(sess,geopath_update_placeholders,geopath_update_ops,geopath_set[second],FLAGS.L,FLAGS.M);
    acc_geo2_tr=0;
    #pathnet.parameters_update(sess,var_update_placeholders,var_update_ops,var_list_backup);
    for j in range(FLAGS.T):
      _, acc_geo2_tmp = sess.run([train_step, accuracy], feed_dict=feed_dict(True,geopath_set[second],int(FLAGS.batch_num/2),int(tr_flag1),int(tr_flag2)));
      tr_flag1=(tr_flag1+FLAGS.batch_num/2)%data_num1;
      tr_flag2=(tr_flag2+FLAGS.batch_num/2)%data_num2;
      acc_geo2_tr+=acc_geo2_tmp;
    #var_list_task2=pathnet.parameters_backup(var_list_to_learn);
    
    # Compatition between two cases
    if(acc_geo1_tr>acc_geo2_tr):
      geopath_set[second]=np.copy(geopath_set[first]);
      geopath_set[second]=pathnet.mutation(geopath_set[second],FLAGS.L,FLAGS.M,FLAGS.N);
      #pathnet.parameters_update(sess,var_update_placeholders,var_update_ops,var_list_task1);
      #train_writer.add_summary(summary_geo1_tr, i);
      print('Training Accuracy at step %s: %s' % (i, acc_geo1_tr/FLAGS.T));
      if(acc_geo1_tr/FLAGS.T >= 0.998):
        print('Learning Done!!');
        print('Optimal Path is as followed.');
        print(geopath_set[first]);
        task1_optimal_path=geopath_set[first];
        break;
    else:
      geopath_set[first]=np.copy(geopath_set[second]);
      geopath_set[first]=pathnet.mutation(geopath_set[first],FLAGS.L,FLAGS.M,FLAGS.N);
      #pathnet.parameters_update(sess,var_update_placeholders,var_update_ops,var_list_task2);
      #train_writer.add_summary(summary_geo2_tr, i);
      print('Training Accuracy at step %s: %s' % (i, acc_geo2_tr/FLAGS.T));
      if(acc_geo2_tr/FLAGS.T >= 0.998):
        print('Learning Done!!');
        print('Optimal Path is as followed.');
        print(geopath_set[second]);
        task1_optimal_path=geopath_set[second];
        break;
        
  iter_task1=i;    
  
  # Fix task1 Optimal Path
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      if(task1_optimal_path[i,j]==1.0):
        fixed_list[i,j]='1';
  
  # Get variables of fixed list
  var_list_to_fix=[];
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
     if(fixed_list[i,j]=='1'):
       var_list_to_fix+=weights_list[i,j]+biases_list[i,j];
  var_list_fix=pathnet.parameters_backup(var_list_to_fix);
 
  # parameters placeholders and ops 
  var_fix_ops=np.zeros(len(var_list_to_fix),dtype=object);
  var_fix_placeholders=np.zeros(len(var_list_to_fix),dtype=object);
  for i in range(len(var_list_to_fix)):
    var_fix_placeholders[i]=tf.placeholder(var_list_to_fix[i].dtype,shape=var_list_to_fix[i].get_shape());
    var_fix_ops[i]=var_list_to_fix[i].assign(var_fix_placeholders[i]);

  ## TASK 2
  # Need to learn variables
  var_list_to_learn=[]+output_weights+output_biases;
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      if (fixed_list[i,j]=='0'):
        var_list_to_learn+=weights_list[i,j]+biases_list[i,j];
 
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      if(fixed_list[i,j]=='1'):
        tmp=biases_list[i,j][0];
        break;
    break;

  # Initialization
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train2', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test2')
  tf.global_variables_initializer().run()
  
  # Update fixed values
  pathnet.parameters_update(sess,var_fix_placeholders,var_fix_ops,var_list_fix);
 
  # GradientDescent  
  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy,var_list=var_list_to_learn);
  
  # Make a TensorFlow feed_dict: maps data onto Tensor placeholders.
  def feed_dict(train,geopaths,batch_num,tr_flag1=0,tr_flag2=0):
    if train or FLAGS.fake_data:
      x_1=tr_data_b1[tr_flag1:tr_flag1+batch_num,:]; x_2=tr_data_b2[tr_flag2:tr_flag2+batch_num,:];
      if(len(x_1)<batch_num):
        x_1=np.append(x_1,tr_data_b1[:(tr_flag1+batch_num)%len(tr_data_b1),:],axis=0);
      if(len(x_2)<batch_num):
        x_2=np.append(x_2,tr_data_b2[:(tr_flag2+batch_num)%len(tr_data_b2),:],axis=0);
      xs=np.append(x_1,x_2,axis=0);
      ys=np.zeros((batch_num*2,2),dtype=float);
      for i in range(len(ys)):
        if(i<batch_num):
          ys[i,0]=1.0;
        else:
          ys[i,1]=1.0;
    return {x: xs, y_: ys,geopath: geopaths}
    
  # Generating randomly geopath
  geopath_set=np.zeros(FLAGS.candi,dtype=object);
  for i in range(FLAGS.candi):
    geopath_set[i]=pathnet.get_geopath(FLAGS.L,FLAGS.M,FLAGS.N);
    
  # parameters placeholders and ops 
  var_update_ops=np.zeros(len(var_list_to_learn),dtype=object);
  var_update_placeholders=np.zeros(len(var_list_to_learn),dtype=object);
  for i in range(len(var_list_to_learn)):
    var_update_placeholders[i]=tf.placeholder(var_list_to_learn[i].dtype,shape=var_list_to_learn[i].get_shape());
    var_update_ops[i]=var_list_to_learn[i].assign(var_update_placeholders[i]);
    
    
  tr_flag1=0; tr_flag2=0; data_num1=len(tr_data_b1); data_num2=len(tr_data_b2); 
  for i in range(FLAGS.max_steps):
    # Select Two Candidate to Tournament 
    first,second=pathnet.select_two_candi(FLAGS.candi);
    
    # First Candidate
    #pathnet.geopath_insert(sess,geopath_update_placeholders,geopath_update_ops,geopath_set[first],FLAGS.L,FLAGS.M);
    acc_geo1_tr=0;
    #var_list_backup=pathnet.parameters_backup(var_list_to_learn);
    #tr_flag_bak=tr_flag;
    for j in range(FLAGS.T):
      _, acc_geo1_tmp = sess.run([train_step,accuracy], feed_dict=feed_dict(True,geopath_set[first],int(FLAGS.batch_num/2),int(tr_flag1),int(tr_flag2)));
      tr_flag1=(tr_flag1+FLAGS.batch_num/2)%data_num1;
      tr_flag2=(tr_flag2+FLAGS.batch_num/2)%data_num2;
      acc_geo1_tr+=acc_geo1_tmp;
    #var_list_task1=pathnet.parameters_backup(var_list_to_learn);
    #tr_flag=tr_flag_bak;
    # Second Candidate
    #pathnet.geopath_insert(sess,geopath_update_placeholders,geopath_update_ops,geopath_set[second],FLAGS.L,FLAGS.M);
    acc_geo2_tr=0;
    #pathnet.parameters_update(sess,var_update_placeholders,var_update_ops,var_list_backup);
    for j in range(FLAGS.T):
      _, acc_geo2_tmp = sess.run([train_step, accuracy], feed_dict=feed_dict(True,geopath_set[second],int(FLAGS.batch_num/2),int(tr_flag1),int(tr_flag2)));
      tr_flag1=(tr_flag1+FLAGS.batch_num/2)%data_num1;
      tr_flag2=(tr_flag2+FLAGS.batch_num/2)%data_num2;
      acc_geo2_tr+=acc_geo2_tmp;
    #var_list_task2=pathnet.parameters_backup(var_list_to_learn);
    
    # Compatition between two cases
    if(acc_geo1_tr>acc_geo2_tr):
      geopath_set[second]=np.copy(geopath_set[first]);
      geopath_set[second]=pathnet.mutation(geopath_set[second],FLAGS.L,FLAGS.M,FLAGS.N);
      #pathnet.parameters_update(sess,var_update_placeholders,var_update_ops,var_list_task1);
      #train_writer.add_summary(summary_geo1_tr, i);
      print('Training Accuracy at step %s: %s' % (i, acc_geo1_tr/FLAGS.T));
      if(acc_geo1_tr/FLAGS.T >= 0.998):
        print('Learning Done!!');
        print('Optimal Path is as followed.');
        print(geopath_set[first]);
        task1_optimal_path=geopath_set[first];
        break;
    else:
      geopath_set[first]=np.copy(geopath_set[second]);
      geopath_set[first]=pathnet.mutation(geopath_set[first],FLAGS.L,FLAGS.M,FLAGS.N);
      #pathnet.parameters_update(sess,var_update_placeholders,var_update_ops,var_list_task2);
      #train_writer.add_summary(summary_geo2_tr, i);
      print('Training Accuracy at step %s: %s' % (i, acc_geo2_tr/FLAGS.T));
      if(acc_geo2_tr/FLAGS.T >= 0.998):
        print('Learning Done!!');
        print('Optimal Path is as followed.');
        print(geopath_set[second]);
        task1_optimal_path=geopath_set[second];
        break;
        
  iter_task2=i;      
  overlap=0;
  for i in range(len(task1_optimal_path)):
    for j in range(len(task1_optimal_path[0])):
      if(task1_optimal_path[i,j]==task2_optimal_path[i,j])&(task1_optimal_path[i,j]==1.0):
        overlap+=1;
  print("Entire Iter:"+str(iter_task1+iter_task2)+",Overlap:"+str(overlap));
  
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Initial learning rate')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/pathnet/logs/binary_mnist',
                      help='Summaries log directory')
  parser.add_argument('--M', type=int, default=10,
                      help='The Number of Modules per Layer')
  parser.add_argument('--L', type=int, default=3,
                      help='The Number of Layers')
  parser.add_argument('--N', type=int, default=3,
                      help='The Number of Selected Modules per Layer')
  parser.add_argument('--T', type=int, default=50,
                      help='The Number of epoch per each geopath')
  parser.add_argument('--batch_num', type=int, default=16,
                      help='The Number of batches per each geopath')
  parser.add_argument('--filt', type=int, default=20,
                      help='The Number of Filters per Module')
  parser.add_argument('--candi', type=int, default=20,
                      help='The Number of Candidates of geopath')
  parser.add_argument('--a1', type=int, default=1,
                      help='The first class of task1')
  parser.add_argument('--a2', type=int, default=2,
                      help='The second class of task1')
  parser.add_argument('--b1', type=int, default=1,
                      help='The first class of task2')
  parser.add_argument('--b2', type=int, default=2,
                      help='The second class of task2')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
