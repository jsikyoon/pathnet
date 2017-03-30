pathnet
===========

Tensorflow Implementation of Pathnet from Google Deepmind.

Implementation is on Tensorflow 1.0

https://arxiv.org/pdf/1701.08734.pdf

"Agents are pathways (views) through the network which determine the subset of parameters that are used and updated by the forwards and backwards passes of the backpropogation algorithm. During learning, a tournament selection genetic algorithm is used to select pathways through the neural network for replication and mutation. Pathway fitness is the performance of that pathway measured according to a cost function. We demonstrate successful transfer learning; fixing the parameters along a path learned on task A and re-evolving a new population of paths for task B, allows task B to be learned faster than it could be learned from scratch or after fine-tuning."
Form Paper

![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/pathnet.PNG)


Binary MNIST classification tasks
-------------------

`
python binary_mnist_pathnet.py 
`

If you want to run that repeatly, then do as followed.

`
./auto_binary_mnist_pathnet.sh
`

# Settings

Basically, almost parameters are used same to paper, however, I used AdamOptimizer with learning_rate=0.001.

Task1 is classification between "5" and "6", and task2 is classification between "6" and "7".

When two candidates are learned & evaluated, learned parameters from each candidates are saved, and winner parameters are updated to model.
In above process, two candidates use same data batchs.

Learning_rate and geopath set are reset after finishing task1.

# Results

For just task 1, saturation speed of pathnet(L=3,M=10,N=3) is slower than selected size network(L=3,N=3). Almost generations for task1 and task2 are just for task1 (over than 90%). In this experiment, "6" class was learned in task1, thus quickly learned in task2.

# failure history

When I did try this experiment, I setted 5 versus 6 for task 1 and 8 versus 9 for task2. 

Then, saturation speed is really really slow impressively. For both tasks, pathnet is slower than selected size network. 

I think in this case, there are no common patterns, thus, when learning task2, parameters are needed tobe re-learned. 

Interesting thing is, in this case also, overlap is 3 or 4 even if parameters of optimal path in task 1 are fixed. That can mean some patterns are commonly used or path with learned parameters is quickly learned, thus, other paths are losed, and removed (just my thinking ^^;;).
