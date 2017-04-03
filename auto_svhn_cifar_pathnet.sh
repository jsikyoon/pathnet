#!/bin/bash

for i in {1..1000}
do
  echo $i "Iteration"
  python svhn_cifar_pathnet.py > svhn_cifar_pathnet.log
  cat svhn_cifar_pathnet.log |tail -1
done
