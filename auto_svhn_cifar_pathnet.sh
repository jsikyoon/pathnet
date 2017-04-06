#!/bin/bash

for i in {1..1000}
do
  python svhn_cifar_pathnet.py > svhn_cifar_pathnet.log
  result=`cat svhn_cifar_pathnet.log |tail -1`
  echo $i "Iteration, " $result
done
