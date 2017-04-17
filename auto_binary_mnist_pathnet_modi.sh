#!/bin/bash

for i in {1..1000}
do
  python binary_mnist_pathnet_modi.py --log_dir /tmp/tensorflow/pathnet/modi > binary_mnist_pathnet_modi.log
  result=`cat binary_mnist_pathnet_modi.log |tail -1`
  echo $i "Iteration, " $result 
done
