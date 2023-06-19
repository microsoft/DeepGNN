# Introduction
__Heterogeneous graphs__ contain abundant information with structural relations (edges) among multi-typed nodes as
well as unstructured content associated with each node.

HetGNN first introduces a random walk with restart strategy to sample a fixed size of strongly correlated heterogeneous neighbors for each node and group them based upon node types. Next, it designs a neural network architecture with two modules to aggregate feature information of those sampled neighboring nodes. The first module encodes “deep” feature interactions of heterogeneous contents and generates content embedding for each node. The second module aggregates content (attribute) embeddings of different neighboring groups (types) and further combines them by considering the impacts of different groups to obtain the ultimate node embedding. Finally, it  leverage a graph context loss and a mini-batch gradient descent procedure to train the model in an end-to-end manner.

Reference: [Heterogeneous Graph Neural Network](https://www3.nd.edu/~dial/publications/zhang_2019_heterogeneous.pdf)
