#ifndef HOEFFDINGTREE_H
#define HOEFFDINGTREE_H


#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#define CLASS_NUMBER 2
#define FEATURE_NUMBER 2
#define MIN_SAMPLES 10
#define R 1.0
#define DELTA 0.5

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