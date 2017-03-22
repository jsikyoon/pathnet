#!/bin/bash

for i in {1..1000}
do
  echo $i "Iteration"
  python binary_mnist_pathnet.py > binary_mnist_pathnet.log
  cat binary_mnist_pathnet.log |tail -1
done
