#include <vector>
#include <numeric>
#include <vector>
#include <algorithm>
#include <ctime>
#include <map>


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
    std::map<int, class_summary> Summary;
    
    public:
    Naive_Bayes(int classNumber){
        for (int i = 0; i < classNumber; i++){
            class_summary summary;
            summary.counter = 1;
            Summary[0] = summary;
        }
    }
    void fit(std::vector<float> newData, int label);
    int  predict(const std::vector<float>& test_data);
};

float prob_By_Summary(const std::vector<float> &test_data, const class_summary &summary );
float calc_prob(float x, float m, float s);
#endif