#include <vector>
#include <cmath>
#include <algorithm>
#include <arduino.h>
#include <ctime>

#include "naiveBayes.h"


void Naive_Bayes::fit(std::vector<std::vector<float>> newData, int label){
    for(std::vector<float> instance : newData){
        //update class summary
        //Serial.print(" ckpt1 ");
        if (Summary[label].counter == 1){
            Summary[label].means = instance;
            Summary[label].sumOfSquares = std::vector<float>(instance.size(),0);
        }else{
            for(unsigned int i = 0; i < Summary[label].means.size(); i++){
                //Serial.print(" ckpt2 ");
                float d1 = instance[i] - Summary[label].means[i];
                Summary[label].means[i] = (Summary[label].means[i] * Summary[label].counter + instance[i]) / (Summary[label].counter + 1);
                float d2 = instance[i] - Summary[label].means[i];
                //Serial.print(" ckpt3 ");
                Summary[label].sumOfSquares[i] += (d1*d2);
            }
        }
        Summary[label].counter++;
    }
    return;
}



int Naive_Bayes::predict(const std::vector<float>& test_data){
    std::vector<float> out;
    for (int label = 0; label < labelNumber; label++){
       out.push_back(prob_By_Summary(test_data ,Summary[label] ));
    }
    int maxElementIndex = std::max_element(out.begin(),out.end()) - out.begin();
    return maxElementIndex;
}


float prob_By_Summary(const std::vector<float> &test_data ,const class_summary &summary ){
    float prob = 1;
    for (unsigned int i = 0; i < summary.means.size(); i++){
        float stdev = sqrt(summary.sumOfSquares[i]/summary.counter);
        prob *= calc_prob(test_data[i],summary.means[i],stdev);
        #ifdef DEBUG
        Serial.print(" stdev: ");
        Serial.print(stdev);
        Serial.print(" mean: ");
        Serial.print(summary.means[i]);
        #endif
    }
    /* multiplying by the class probability*/
    prob *= summary.class_prob;
    return prob;
}

float calc_prob(float x, float m, float s)
{
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) / s;

    return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}