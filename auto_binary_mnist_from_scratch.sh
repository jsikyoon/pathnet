#!/bin/bash

for i in {1..1000}
do
  python binary_mnist_from_scratch.py > binary_mnist_scratch.log
  result=`cat binary_mnist_scratch.log |tail -1`
  echo $i "Iteration, " $result 
done
