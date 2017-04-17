#!/bin/bash

for i in {1..1000}
do
  python binary_mnist_fine_tunning.py > binary_mnist_fine_tunning.log
  result=`cat binary_mnist_fine_tunning.log |tail -1`
  echo $i "Iteration, " $result 
done
