from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn
import tensorflow as tf
import tflearn.datasets.mnist as mnist
import sys

def fc_module(net,filt):
  return tflearn.fully_connected(net, filt, activation='relu',regularizer='L2', weight_decay=0.001);

if __name__=="__main__":
  epoch=10
  dataset="mnist"
  width=28
  height=28
  channel=1
  class_num=10
  filt=64;
  M=2;L=1;geopath=np.zeros((L,M),dtype=int)+1;
  X, Y, testX, testY = mnist.load_data(one_hot=True)
  X = X.reshape([-1, 28, 28, 1])
  testX = testX.reshape([-1, 28, 28, 1])
  net = tflearn.input_data(shape=[None, width, height, channel], name='input')
  #Input Layer
  net = tflearn.fully_connected(net, filt, activation='relu',
                            regularizer='L2', weight_decay=0.001);	
							
  #Hidden Layers
  for l in range(L):							
    #1st Layer of Pathnet
    net_list=np.zeros(M,dtype=object);
    for m in range(M):
      net_list[m]=fc_module(net,filt); 
    #Between Layers activation is summed
    net=np.sum(net_list[geopath[l,:]==1]);

  #Output Layer
  net = tflearn.fully_connected(net, class_num, activation='softmax');
  mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
  net = tflearn.regression(net, optimizer=mom,
                           loss='categorical_crossentropy');
  print(net);
  """
  #Training
  model = tflearn.DNN(net, checkpoint_path=('models/model_pathnet_'+dataset),
                      max_checkpoints=10, tensorboard_verbose=0,
                      clip_gradients=0.);
  model.fit(X, Y, n_epoch=epoch, validation_set=(testX, testY),
            snapshot_epoch=False, snapshot_step=500,
            show_metric=True, batch_size=128, shuffle=True,
            run_id=('pathnet_'+dataset));
  #Testing
  aa=model.predict(testX);
  correct=0;
  for i in range(len(aa)):
    if(aa[i].index(max(aa[i])) == np.argmax(testY[i])):
      correct=correct+1;
  """