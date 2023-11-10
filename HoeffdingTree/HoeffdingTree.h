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
    ClassObserver(std::vector<float> means, std::vector<float> sumOfSquares, std::vector<float> maxValues, std::vector<float> minValues, int counter) :
        means(means),
        sumOfSquares(sumOfSquares),
        maxValues(maxValues),
        minValues(minValues),
        counter(counter) {}
    ClassObserver(int counter) : counter(counter) {
        means = std::vector<float>();
        sumOfSquares = std::vector<float>();
        maxValues = std::vector<float>();
        minValues = std::vector<float>();
    }
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
    int min_samples = 50;
    unsigned int number_of_classes;
    std::map<int, ClassObserver> classes;
    std::vector<Node> children;
    float gini_index_of_split(Split split);
    void update_statistics(const std::vector<float>& features, int class_label);
    std::vector<float> getSplitSuggestions(int feature);
    std::vector<Split> getBestSplitSuggestions();
    void attemptToSplit();

    Node(unsigned int number_of_classes) : is_leaf(true), feature(-1), split_value(0), class_label(0), number_of_classes(number_of_classes) {
        classes = std::map<int, ClassObserver>();
        for (int i = 0; i < number_of_classes; i++){
            classes[i] = ClassObserver();
        }
    }
    Node(bool is_leaf, int feature, float split_value, int class_label, std::map<int, ClassObserver> classes, std::vector<Node> children, unsigned int number_of_classes) :
        is_leaf(is_leaf),
        feature(feature),
        split_value(split_value),
        class_label(class_label),
        classes(classes),
        children(children),
        number_of_classes(number_of_classes) {}
};


//float calc_prob(float x, float m, float s);
//float erf_x(float x);

class HoeffdingTree {
    public:
        Node root;
        float delta;
        unsigned int min_samples;
        unsigned int number_of_classes;

        HoeffdingTree(float delta, int min_samples, unsigned int number_of_classes, Node root)
                : delta(delta), min_samples(min_samples), number_of_classes(number_of_classes), root(root) {
            root = Node(number_of_classes);
        }

        void fit(const std::vector<float>&features, int class_label);

        int predict(const std::vector<float>& features);

        HoeffdingTree(Node root, float delta, int min_samples, unsigned int number_of_classes) :
            root(root),
            delta(delta),
            min_samples(min_samples),
            number_of_classes(number_of_classes) {}
};
#endif