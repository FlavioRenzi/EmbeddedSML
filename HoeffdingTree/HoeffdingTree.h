#ifndef HOEFFDINGTREE_H
#define HOEFFDINGTREE_H


#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include "Config.h"

typedef struct FeatureObserver{
    float mean;
    float sum_of_squares;
    float max_value;
    float min_value;
} FeatureObserver;

typedef struct ClassObserver{
    FeatureObserver feature_observers[FEATURE_NUMBER];
    float counter;
} ClassObserver;

typedef struct Node{
    int is_leaf;
    int feature;
    float split_value;
    ClassObserver class_observers[CLASS_NUMBER];
    struct Node* children[2];
} Node;

int predict(Node* tree, float* features);
void fit(Node* tree, float* features, int class_label);

#endif