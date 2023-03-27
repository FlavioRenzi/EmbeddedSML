#include <vector>
#include <numeric>
#include <vector>
#include <algorithm>
#include <ctime>


#ifndef NAIVEBAYES_H
#define NAIVEBAYES_H

typedef struct class_summary {  
    std::vector<float> means;
    std::vector<float> sumOfSquares;
    int counter;
    float class_prob;

} class_summary; 

class Naive_Bayes{
    private:
    std::vector<class_summary> Summary;
    int labelNumber;
    
    public:
    Naive_Bayes(int classNumber){
        labelNumber = classNumber;
        for (int i = 0; i < classNumber; i++){
            class_summary summary;
            summary.counter = 1;
            summary.class_prob = 1.0/classNumber;
            Summary.push_back(summary);
        }
    }
    void fit(std::vector<std::vector<float>> newData, int label);
    int  predict(const std::vector<float>& test_data);
};

float prob_By_Summary(const std::vector<float> &test_data, const class_summary &summary );
float calc_prob(float x, float m, float s);
#endif