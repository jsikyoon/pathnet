# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pathnet


# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
  def __init__(self,
               action_size,
               thread_index, # -1 for global               
               device="/cpu:0"):
    self._action_size = action_size
    self._thread_index = thread_index
    self._device = device    

  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])

      # avoid NaN with clipping when value in pi becomes zero
      log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))
      
      # policy entropy
      entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)
      
      # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
      policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.a ), reduction_indices=1 ) * self.td + entropy * entropy_beta )

      # R (input for value)
      self.r = tf.placeholder("float", [None])
      
      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

      # gradienet of policy and value are summed up
      self.total_loss = policy_loss + value_loss

  def run_policy_and_value(self, sess, s_t):
    raise NotImplementedError()
    
  def run_policy(self, sess, s_t):
    raise NotImplementedError()

  def run_value(self, sess, s_t):
    raise NotImplementedError()    

  def get_vars(self):
    raise NotImplementedError()

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "GameACNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_variable(self, weight_shape):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv_variable(self, weight_shape):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0"):
    GameACNetwork.__init__(self, action_size, thread_index, device)

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
      self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32]) # stride=2

      self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256])

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

      # state (input)
      self.s = tf.placeholder("float", [None, 84, 84, 4])
    
      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

      h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

      # policy (output)
      self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
      # value (output)
      v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )

  def run_policy_and_value(self, sess, s_t):
    pi_out, v_out = sess.run( [self.pi, self.v], feed_dict = {self.s : [s_t]} )
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
    return pi_out[0]

  def run_value(self, sess, s_t):
    v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]

# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    GameACNetwork.__init__(self, action_size, thread_index, device)

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
      self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32]) # stride=2
      
      self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256])

      # lstm
      self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

      # state (input)
      self.s = tf.placeholder("float", [None, 84, 84, 4])
    
      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

      h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
      # h_fc1 shape=(5,256)

      h_fc1_reshaped = tf.reshape(h_fc1, [1,-1,256])
      # h_fc_reshaped = (1,5,256)

      # place holder for LSTM unrolling time step size.
      self.step_size = tf.placeholder(tf.float32, [1])

      self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
      self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
      self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                              self.initial_lstm_state1)
      
      # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
      # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
      # Unrolling step size is applied via self.step_size placeholder.
      # When forward propagating, step_size is 1.
      # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
      lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                        h_fc1_reshaped,
                                                        initial_state = self.initial_lstm_state,
                                                        sequence_length = self.step_size,
                                                        time_major = False,
                                                        scope = scope)

      # lstm_outputs: (1,5,256) for back prop, (1,1,256) for forward prop.
      
      lstm_outputs = tf.reshape(lstm_outputs, [-1,256])

      # policy (output)
      self.pi = tf.nn.softmax(tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2)
      
      # value (output)
      v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )

      scope.reuse_variables()
      self.W_lstm = tf.get_variable("basic_lstm_cell/weights")
      self.b_lstm = tf.get_variable("basic_lstm_cell/biases")

      self.reset_state()
      
  def reset_state(self):
    self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
                                                        np.zeros([1, 256]))

  def run_policy_and_value(self, sess, s_t):
    # This run_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    pi_out, v_out, self.lstm_state_out = sess.run( [self.pi, self.v, self.lstm_state],
                                                   feed_dict = {self.s : [s_t],
                                                                self.initial_lstm_state0 : self.lstm_state_out[0],
                                                                self.initial_lstm_state1 : self.lstm_state_out[1],
                                                                self.step_size : [1]} )
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    # This run_policy() is used for displaying the result with display tool.    
    pi_out, self.lstm_state_out = sess.run( [self.pi, self.lstm_state],
                                            feed_dict = {self.s : [s_t],
                                                         self.initial_lstm_state0 : self.lstm_state_out[0],
                                                         self.initial_lstm_state1 : self.lstm_state_out[1],
                                                         self.step_size : [1]} )
                                            
    return pi_out[0]

  def run_value(self, sess, s_t):
    # This run_value() is used for calculating V for bootstrapping at the 
    # end of LOCAL_T_MAX time step sequence.
    # When next sequcen starts, V will be calculated again with the same state using updated network weights,
    # so we don't update LSTM state here.
    prev_lstm_state_out = self.lstm_state_out
    v_out, _ = sess.run( [self.v, self.lstm_state],
                         feed_dict = {self.s : [s_t],
                                      self.initial_lstm_state0 : self.lstm_state_out[0],
                                      self.initial_lstm_state1 : self.lstm_state_out[1],
                                      self.step_size : [1]} )
    
    # roll back lstm state
    self.lstm_state_out = prev_lstm_state_out
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_lstm, self.b_lstm,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]
            
# Actor-Critic PathNet Network
class GameACPathNetNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0"
               ,FLAGS=""):
    GameACNetwork.__init__(self, action_size, thread_index, device)

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      # First three Layers
      self.W_conv=np.zeros((FLAGS.L-1,FLAGS.M),dtype=object);
      self.b_conv=np.zeros((FLAGS.L-1,FLAGS.M),dtype=object);
      kernel_num=np.array(FLAGS.kernel_num.split(","),dtype=int);
      stride_size=np.array(FLAGS.stride_size.split(","),dtype=int);
      feature_num=[8,8,8];
      last_lin_num=392;
      #last_lin_num=2592;
      for i in range(FLAGS.L-1):
        for j in range(FLAGS.M):
          if(i==0):
            self.W_conv[i,j], self.b_conv[i,j] = self._conv_variable([kernel_num[i],kernel_num[i],4,feature_num[i]]);
          else:
            self.W_conv[i,j], self.b_conv[i,j] = self._conv_variable([kernel_num[i],kernel_num[i],feature_num[i-1],feature_num[i]]);
      
      # Last Layer in PathNet      
      self.W_lin=np.zeros(FLAGS.M,dtype=object);
      self.b_lin=np.zeros(FLAGS.M,dtype=object);
      for i in range(FLAGS.M):
        self.W_lin[i], self.b_lin[i] = self._fc_variable([last_lin_num, 256])

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])
  
      # geopath_examples
      self.geopath=pathnet.geopath_initializer(FLAGS.L,FLAGS.M);
 
      # geopathes placeholders and ops
      self.geopath_update_ops=np.zeros((len(self.geopath),len(self.geopath[0])),dtype=object);
      self.geopath_update_placeholders=np.zeros((len(self.geopath),len(self.geopath[0])),dtype=object);
      for i in range(len(self.geopath)):
        for j in range(len(self.geopath[0])):
          self.geopath_update_placeholders[i,j]=tf.placeholder(self.geopath[i,j].dtype,shape=self.geopath[i,j].get_shape());
          self.geopath_update_ops[i,j]=self.geopath[i,j].assign(self.geopath_update_placeholders[i,j]);

 
      # fixed weights list
      self.fixed_path=np.ones((FLAGS.L,FLAGS.M),dtype=str);
      for i in range(FLAGS.L):
        for j in range(FLAGS.M):
          self.fixed_path[i,j]='0';    
      
      # state (input)
      self.s = tf.placeholder("float", [None, 84, 84, 4])
      
      for i in range(FLAGS.L):
        layer_modules_list=np.zeros(FLAGS.M,dtype=object);
        if(i==FLAGS.L-1):
          net=tf.reshape(net,[-1,last_lin_num]);
        for j in range(FLAGS.M):
          if(i==0):
            layer_modules_list[j]=tf.nn.relu(self._conv2d(self.s,self.W_conv[i,j],stride_size[i])+self.b_conv[i,j])*self.geopath[i,j];
          elif(i==FLAGS.L-1):
            layer_modules_list[j]=tf.nn.relu(tf.matmul(net,self.W_lin[j])+self.b_lin[j])*self.geopath[i,j];
          else:
            layer_modules_list[j]=tf.nn.relu(self._conv2d(net,self.W_conv[i,j],stride_size[i])+self.b_conv[i,j])*self.geopath[i,j];
        net=np.sum(layer_modules_list);
      net=net/FLAGS.M;  
      # policy (output)
      self.pi = tf.nn.softmax(tf.matmul(net, self.W_fc2) + self.b_fc2)
      # value (output)
      v_ = tf.matmul(net, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )

  def run_policy_and_value(self, sess, s_t):
    pi_out, v_out = sess.run( [self.pi, self.v], feed_dict = {self.s : [s_t]} )
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
    return pi_out[0]

  def run_value(self, sess, s_t):
    v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
    return v_out[0]

  def get_vars(self):
    res=[];
    for i in range(len(self.W_conv)):
      for j in range(len(self.W_conv[0])):
        if(self.fixed_path[i,j]=='0'):
          res+=[self.W_conv[i,j]]+[self.b_conv[i,j]];
    for i in range(len(self.W_lin)):
      if(self.fixed_path[-1,j]=='0'):
        res+=[self.W_lin[i]]+[self.b_lin[i]];
    res+=[self.W_fc2]+[self.b_fc2];
    res+=[self.W_fc3]+[self.b_fc3];
    return res;
  
  def get_vars2(self):
    res=[];
    for i in range(len(self.W_conv)):
      for j in range(len(self.W_conv[0])):
        if(self.fixed_path[i,j]=='1'):
          res+=[self.W_conv[i,j]]+[self.b_conv[i,j]];
    for i in range(len(self.W_lin)):
      if(self.fixed_path[-1,j]=='1'):
        res+=[self.W_lin[i]]+[self.b_lin[i]];
    #res+=[self.W_fc2]+[self.b_fc2];
    #res+=[self.W_fc3]+[self.b_fc3];
    return res;
  
  def get_vars_idx(self):
    res=[];
    for i in range(len(self.W_conv)):
      for j in range(len(self.W_conv[0])):
        if(self.fixed_path[i,j]=='0'):
          res+=[1,1];
        else:
          res+=[0,0];
    for i in range(len(self.W_lin)):
      if(self.fixed_path[-1,j]=='0'):
        res+=[1,1];
      else:
        res+=[0,0];
    res+=[1,1];
    res+=[1,1];
    return res;
