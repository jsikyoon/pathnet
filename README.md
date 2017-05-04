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
L, M, N, B and the number of populations are 3, 10, 3, 2 and 20, respectively (In paper, the number of populations is 64.). 
GradientDescent Method is used with learning rate=0.05 (In paper, learning rate=0.0001.).
Aggregation function between layers is average (In paper, that is summation.).
Skip connection, Resnet and linear modules are used for each layers except input layer.
Fixed path of first task is always activated when feed-forwarding the networks on second task (In paper, the path is not always activated.).

Chrisantha Fernando (1st author of this paper)  and I checked the results of the paper was generated when the value is 20. Thus, I set that as 20.
I set bigger learning rate vaule than that of paper for getting results faster than before.
Higher learning rate can accelate network learning faster than positive transfer learning. For de-accelating converage, average function is used.
The author and I checked the paper results was generated when last aggregation function is average not summation (Except last one, others are summation.).
Fixed path activation is for generating more dramatic results than before.

B candidates use same data batchs.
geopath set and parameters except the ones on optimal path of first task are reset after finishing first task.


### Results
![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/binary_mnist_1vs3_1vs2.PNG) 
![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/binary_mnist_6vs7_4vs5.PNG) 
![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/binary_mnist_4vs5_graph.PNG) 

The experiments were 1vs3 <-> 1vs2 and 4vs5 <-> 6vs7. 
The reason of selecting those classes is to check positive transfer learning whenever there are sharing class or not. 

1vs3 experiments showed first task and second task after 1vs2 converage generation means are about 169.515 and 83.2. 
Pathnet made about 2 times faster converage than that from the scratch.

1vs2 experiments showed first task and second task after 1vs3 converage generation means are about 193.67 and 116.615. 
Pathnet made about 1.7 times faster converage than that from the scratch.

4vs5 experiments showed first task and second task after 6vs7 converage generation means are about 260.195 and 147.55. 
Pathnet made about 1.8 times faster converage than that from the scratch.

6vs7 experiments showed first task and second task after 4vs5 converage generation means are about 93.97 and 55.23. 
Pathnet made about 1.7 times faster converage than that from the scratch.

Pathnet showed about 1.7~2 times better performance than that of "learning from scratch" on Binary MNIST Classification whenever there are sharing class or not.
