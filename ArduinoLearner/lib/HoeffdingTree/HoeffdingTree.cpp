
#include "hoeffdingTree.h"

float ClassObserver::variance(int class_label) {
    return (sumOfSquares[class_label]/counter);
}

void ClassObserver::addInstance(std::vector<float> features){
    if (counter == 0){
        means = features;
        maxValues = features;
        minValues = features;
        sumOfSquares = std::vector<float>(features.size(),0);
    }else{
        for(unsigned int i = 0; i < means.size(); i++){
            float d1 = features[i] - means[i];
            means[i] = (means[i] * counter + features[i]) / (counter + 1);
            float d2 = features[i] - means[i];
            sumOfSquares[i] += (d1*d2);

            if (features[i] > maxValues[i]){
                maxValues[i] = features[i];
            }
            if (features[i] < minValues[i]){
                minValues[i] = features[i];
            }
        }
    }
    counter++;
}

float ClassObserver::getFeatureProbability(int featureNumber, float x){
    if (counter == 0 || sumOfSquares[featureNumber] == 0){
        return 0;
    }
    float std_deviation = sqrt(sumOfSquares[featureNumber]/counter);
    //Serialdbg.println("std_deviation: " + String(std_deviation) + ", means[featureNumber]: " + String(means[featureNumber]) + ", x: " + String(x));
    float z = (x - means[featureNumber]) / std_deviation;
    //Serialdbg.println("z: " + String(z));
    float p = (1.0 + std::erf(z / std::sqrt(2.0))) / 2.0;
    //Serialdbg.println("p: " + String(p));
    return p;
}

void Node::update_statistics(const std::vector<float>& features, int class_label){
    if (is_leaf){
        if (classes.find(class_label) == classes.end()){
            classes[class_label] = ClassObserver();
        }
        classes[class_label].addInstance(features);
        int bestClass = -1;
        int max=-1;
        for (auto& classObserver : classes){
            if (classObserver.second.counter > max){
                max = classObserver.second.counter;
                bestClass = classObserver.first;
            }
        }
        this->class_label = 1-bestClass;
    }else{
        if (features[feature] < split_value){
            children[0]->update_statistics(features, class_label);
        }else{
            children[1]->update_statistics(features, class_label);
        }
    }
}

std::vector<Split> Node::getBestSplitSuggestions(){
    std::vector<Split> bestSplits;
    for (unsigned int i = 0; i < classes.size(); i++){
        Split split;
        for (unsigned int feature = 0; feature < classes[i].means.size(); feature++){
            //Serialdbg.println("getting split suggestions");
            std::vector<float> splitSuggestions = getSplitSuggestions(feature);
            //Serialdbg.println("got split suggestions");
            for (auto& splitValue : splitSuggestions){
                split.feature = feature;
                split.split_value = splitValue;
                split.left_classes = std::vector<ClassObserver>(classes.size());
                split.right_classes = std::vector<ClassObserver>(classes.size());
                split.left_dist = std::vector<float>(classes.size());
                split.right_dist = std::vector<float>(classes.size());
                
                for (unsigned int j = 0; j < classes.size(); j++){
                    //Serialdbg.println("adding instance to split");
                    split.left_dist[j] = classes[j].getFeatureProbability(feature, splitValue);
                    //Serialdbg.println("added instance to split");
                    split.right_dist[j] = 1- split.left_dist[j];
                    //Serialdbg.println("added instance to split");
                }
                //split.left_classes[i].maxValues[feature] = splitValue;
                //split.right_classes[i].minValues[feature] = splitValue;
                //Serialdbg.println("calculating gini");
                split.gini = gini_index_of_split(split);

                bestSplits.push_back(split);
            }
        }
    }
    return bestSplits;
}
float computeGini(std::vector<float> dist, float weight){
    float gini = 1;
    for (unsigned int i = 0; i < dist.size(); i++){
        float real = dist[i] / weight;
        gini -= real * real;
    }
    return gini;
}

float Node::gini_index_of_split(Split split){
    float left_weight = 0;
    float right_weight = 0;
    for (unsigned int i = 0; i < split.left_dist.size(); i++){
        left_weight += split.left_dist[i];
        right_weight += split.right_dist[i];
    }
    //Serialdbg.println("left weight: " + String(left_weight) + ", right weight: " + String(right_weight));

    float gini = 0.0;
    gini += (left_weight/(left_weight + right_weight)) * computeGini(split.left_dist, left_weight);
    
    gini += (right_weight/(left_weight + right_weight)) * computeGini(split.right_dist, right_weight);
    
    return 1.0 - gini;
}

std::vector<float> Node::getSplitSuggestions(int feature){
    std::vector<float> splitSuggestions;
    float max_value = std::numeric_limits<float>::min();
    float min_value = std::numeric_limits<float>::max();
    for (auto& classObserver : classes){
        if (classObserver.second.maxValues[feature] > max_value){
            max_value = classObserver.second.maxValues[feature];
        }
        if (classObserver.second.minValues[feature] < min_value){
            min_value = classObserver.second.minValues[feature];
        }
    }
    int num_splits = 10; //todo make this a parameter
    for (int i = 1; i <= num_splits; i++){
        float splitValue = min_value + (max_value - min_value) * i / num_splits;
        if (splitValue > min_value && splitValue < max_value){
            splitSuggestions.push_back(splitValue);
        }
    }
    return splitSuggestions;
}

void Node::attemptToSplit(){
    int min_samples = 10; //todo make this a parameter
    int num_samples = std::accumulate(classes.begin(), classes.end(), 0, [](int sum, std::pair<int, ClassObserver> p){return sum + p.second.counter;});
    if (num_samples < min_samples){
        return;
    }
    //todo add check for purity
    std::vector<Split> bestSplits = getBestSplitSuggestions();
    //Serialdbg.println("best splits");
    float bestGini = std::numeric_limits<float>::min();
    float secondBestGini = std::numeric_limits<float>::min();
    Split bestSplit;
    Split secondBestSplit;
    for (auto& split : bestSplits){
        //Serialdbg.println("gini: " + String(split.gini));
        if (split.gini > bestGini){
            secondBestGini = bestGini;
            secondBestSplit = bestSplit;
            bestGini = split.gini;
            bestSplit = split;
        }else if (split.gini > secondBestGini && split.gini != bestGini){
            secondBestGini = split.gini;
            secondBestSplit = split;
        }
    }
    //Serialdbg.println("best gini: " + String(bestGini));
    //compute Hoeffding bound
    float R = 1.0;
    float delta = 0.5;
    float n = std::accumulate(classes.begin(), classes.end(), 0, [](int sum, std::pair<int, ClassObserver> p){return sum + p.second.counter;});

    float epsilon = sqrt((R*R*log(1.0/delta))/(2.0*n));
    //Serialdbg.println("epsilon: " + String(epsilon));
    Serialdbg.println("best gini: " + String(bestGini) + ", second best gini: " + String(secondBestGini) + ", epsilon: " + String(epsilon));

    if (bestGini - secondBestGini > epsilon){
        //split
        Serialdbg.println("splitting");
        is_leaf = false;
        feature = bestSplit.feature;
        split_value = bestSplit.split_value;
        class_label = 1;
        children.push_back(new Node());
        children.push_back(new Node());
        for (unsigned int  i = 0; i < classes.size(); i++){
            children[0]->classes[i] = bestSplit.left_classes[i];
            children[1]->classes[i] = bestSplit.right_classes[i];
        }
    }

}

void HoeffdingTree::fit(const std::vector<float>&features, int class_label){
    Node* node = root;
    while (!node->is_leaf){
        if (features[node->feature] < node->split_value){
            node = node->children[0];
        }else{
            node = node->children[1];
        }
    }
    //Serialdbg.println("updating statistics");
    node->update_statistics(features, class_label);
    //Serialdbg.println("attempting to split");
    node->attemptToSplit();
}

int HoeffdingTree::predict(const std::vector<float>&features){
    Node* node = root;
    while (!node->is_leaf){
        if (features[node->feature] < node->split_value){
            node = node->children[0];
        }else{
            node = node->children[1];
        }
    }
    //Serialdbg.println("predicted class: "+ String(node->class_label));
    return node->class_label;
}