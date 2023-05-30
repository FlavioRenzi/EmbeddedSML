### ClassObserver

contains the summary of a single class to obtain the distribution

### Split

struct to store a possible split

### Node

class to generate nodes of the tree
- vector of children
- a map to store the distribution of each classes
- the feature to split on
- the threshold to split on

### HoeffedingTree

it contains the root and the functions to fit and predict from the tree

## Splitting method
- a new instance arrives and teh fit function is invoked
- the instance is filtered in the existing tree to reach a leaf
- the leaf is updated with the new instance
- we attempt to split the leaf
- we calculate the split suggustions
- 