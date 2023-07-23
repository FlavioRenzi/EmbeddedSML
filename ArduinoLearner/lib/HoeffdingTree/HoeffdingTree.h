#include <iostream>
#include <vector>
#include <map>
#include <vector>
#include <map>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <queue>

#include <Arduino.h>




#ifndef HOEFFDINGTREE_H
#define HOEFFDINGTREE_H



class ClassObserver{ 
public: 
    std::vector<float> means;
    std::vector<float> sumOfSquares;
    std::vector<float> maxValues;
    std::vector<float> minValues;
    int counter;

    float getFeatureProbability(int featureNumber, float x);
    void addInstance(std::vector<float> features);
    float variance(int class_label);

    ClassObserver() : counter(0) {}
};

typedef struct Split {  
    int feature;
    float split_value;
    float gini;
    int left_class;
    std::vector<ClassObserver> left_classes;
    std::vector<float> left_dist;
    int right_class;
    std::vector<ClassObserver> right_classes;
    std::vector<float> right_dist;
} Split; 


class Node {
public:
    bool is_leaf;
    int feature;
    float split_value;
    int class_label;
    std::map<int, ClassObserver> classes;
    std::vector<Node*> children;
    float gini_index_of_split(Split split);
    void update_statistics(const std::vector<float>& features, int class_label);
    std::vector<float> getSplitSuggestions(int feature);
    std::vector<Split> getBestSplitSuggestions();
    void attemptToSplit();

    Node() : is_leaf(true), feature(-1), split_value(0), class_label(0) {}
    

};


//float calc_prob(float x, float m, float s);
//float erf_x(float x);

class HoeffdingTree {
    public:
        Node* root;
        float delta;
        unsigned int min_samples;

        HoeffdingTree(float delta, int min_samples) : delta(delta), min_samples(min_samples) {
            root = new Node();
        }

        void fit(const std::vector<float>&features, int class_label);

        int predict(const std::vector<float>& features);
        
};
#endif