pathnet
===========

Tensorflow Implementation of Pathnet from Google Deepmind.

Implementation is on Tensorflow 1.0

https://arxiv.org/pdf/1701.08734.pdf

"Agents are pathways (views) through the network which determine the subset of parameters that are used and updated by the forwards and backwards passes of the backpropogation algorithm. During learning, a tournament selection genetic algorithm is used to select pathways through the neural network for replication and mutation. Pathway fitness is the performance of that pathway measured according to a cost function. We demonstrate successful transfer learning; fixing the parameters along a path learned on task A and re-evolving a new population of paths for task B, allows task B to be learned faster than it could be learned from scratch or after fine-tuning."
Form Paper

![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/pathnet.PNG)

### Failure Story

Memory Leak Problem was happened without placeholder for geopath. Without placeholder, changing the value of tensor variable is to assign new memory, thus assigning new path for each generation caused memory leak and slow learning.

Binary MNIST classification tasks
-------------------

`
python binary_mnist_pathnet.py 
`

If you want to run that repeatly, then do as followed.

`
./auto_binary_mnist_pathnet.sh
`

### Settings

Basically, almost parameters are used same to paper, however, I used learning_rate=0.01.

When two candidates are learned & evaluated (evaluation is based on training data accuracy), learned parameters from each candidates are saved, and winner parameters are updated to model.
In above process, two candidates use same data batchs.
geopath set are reset after finishing task1.

### Results
![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/binary_mnist_1vs3_1vs2.PNG) 
![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/binary_mnist_6vs7_4vs5.PNG) 

### Failure Story

