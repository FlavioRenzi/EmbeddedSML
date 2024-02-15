#include "HoeffdingTree.h"


float calculateGiniForSplit(Node* node, int feature, float split_value);
void attemptToSplit(Node* node);
float getFeatureProbability(ClassObserver classObserver,int featureNumber, float x);
void update_statistics(Node* node, float* features, int class_label);

void update_statistics(Node* node, float* features, int class_label){
    if (node->is_leaf) {
        if (class_label >= CLASS_NUMBER) {
            //error
            return;
        }
        if (node->class_observers[class_label].counter == 0){
            for (int i = 0; i < FEATURE_NUMBER; i++){
                node->class_observers[class_label].feature_observers[i].mean = features[i];
                node->class_observers[class_label].feature_observers[i].sum_of_squares = 0;
                node->class_observers[class_label].feature_observers[i].max_value = features[i];
                node->class_observers[class_label].feature_observers[i].min_value = features[i];
            }
        } else {
            for (int i = 0; i < FEATURE_NUMBER; i++){
                float d1 = features[i] - node->class_observers[class_label].feature_observers[i].mean;
                node->class_observers[class_label].feature_observers[i].mean = (node->class_observers[class_label].feature_observers[i].mean * node->class_observers[class_label].counter + features[i]) / (node->class_observers[class_label].counter + 1);
                float d2 = features[i] - node->class_observers[class_label].feature_observers[i].mean;
                node->class_observers[class_label].feature_observers[i].sum_of_squares += (d1*d2);

                node->class_observers[class_label].feature_observers[i].max_value = fmax(node->class_observers[class_label].feature_observers[i].max_value, features[i]);
                node->class_observers[class_label].feature_observers[i].min_value = fmin(node->class_observers[class_label].feature_observers[i].min_value, features[i]);
            }
        }
        node->class_observers[class_label].counter++;
    }
}

float getFeatureProbability(ClassObserver classObserver,int featureNumber, float x){
    if (classObserver.counter == 0 || classObserver.feature_observers[featureNumber].sum_of_squares == 0){
        return 0;
    }
    float std_deviation = sqrt(classObserver.feature_observers[featureNumber].sum_of_squares/classObserver.counter);
    //Serialdbg.println("std_deviation: " + String(std_deviation) + ", means[featureNumber]: " + String(means[featureNumber]) + ", x: " + String(x));
    float z = (x - classObserver.feature_observers[featureNumber].mean) / std_deviation;
    //Serialdbg.println("z: " + String(z));
    float p = (1.0 + erf(z / sqrt(2.0))) / 2.0;
    //Serialdbg.println("p: " + String(p));
    return p;
}

float calculateGiniForSplit(Node* node, int feature, float split_value) {
    // Pseudo-code for calculating Gini impurity of a split
    // You need to determine how to calculate this based on your data structure
    float left_dist[CLASS_NUMBER] = {0};
    float right_dist[CLASS_NUMBER] = {0};
    float left_sum = 0, right_sum = 0;

    for (int i = 0; i < CLASS_NUMBER; i++) {
        left_dist[i] = getFeatureProbability(node->class_observers[i],feature,split_value);
        right_dist[i] = 1 - left_dist[i];
    }

    for (int i = 0; i < CLASS_NUMBER; i++) {
        left_sum += left_dist[i];
        right_sum += right_dist[i];
    }
    float split_gini = 0;
    for (int i = 0; i < CLASS_NUMBER; i++) {
        float gini_left = 1;
        float gini_right = 1;
        for (int j = 0; j < CLASS_NUMBER; j++) {
            gini_left -= pow(left_dist[j]/left_sum, 2);
            gini_right -= pow(right_dist[j]/right_sum, 2);
        }
        split_gini += (left_sum/(left_sum+right_sum)) * gini_left;
        split_gini += (right_sum/(left_sum+right_sum)) * gini_right;
    }
    return 1 - split_gini;
}



void attemptToSplit(Node* node){
    float sum = 0;
    for (int i = 0; i < CLASS_NUMBER; i++){
        sum += node->class_observers[i].counter;
    }
    if (sum < MIN_SAMPLES){
        return;
    }
    float best_gini = FLT_MIN;
    float second_best_gini = FLT_MIN;
    int best_feature = -1;
    float best_split_value = 0;
    //find the best split
    for (int feature = 0; feature < FEATURE_NUMBER; feature++) {

        float min_value = FLT_MAX;
        float max_value = FLT_MIN;
        for (int i = 0; i < CLASS_NUMBER; i++){
            min_value = fmin(min_value, node->class_observers[i].feature_observers[feature].min_value);
            max_value = fmax(max_value, node->class_observers[i].feature_observers[feature].max_value);
        }

        float possible_split_values_for_feature[10];

        for(int i = 0; i < 10; i++) {
            possible_split_values_for_feature[i] = min_value + (max_value - min_value) * i / 10;
        }

        for (int i = 0; i < 10; i++) {
            float split_value = possible_split_values_for_feature[i];
            float gini = calculateGiniForSplit(node, feature, split_value);
            if (gini > best_gini) {
                second_best_gini = best_gini;
                best_gini = gini;
                best_feature = feature;
                best_split_value = split_value;
            } else if (gini > second_best_gini) {
                second_best_gini = gini;
            }
        }
    }

    //check the Hoeffding bound
    float epsilon = sqrt((R*R*log(1.0/DELTA))/(2.0*sum));
    if (best_gini - second_best_gini > epsilon){
        //splitting
        //printf("Splitting with feature %d and split value %f\n", best_feature, best_split_value);
        node->is_leaf = 0;
        node->feature = best_feature;
        node->split_value = best_split_value;
        node->children[0] = (Node*)malloc(sizeof(Node));
        node->children[1] = (Node*)malloc(sizeof(Node));
        node->children[0]->is_leaf = 1;
        node->children[1]->is_leaf = 1;
        for (int class = 0; class < CLASS_NUMBER; class++){
            node->children[0]->class_observers[class].counter = 0;
            node->children[1]->class_observers[class].counter = 0;
            for (int feature = 0; feature < FEATURE_NUMBER; feature++){
                node->children[0]->class_observers[class].feature_observers[feature] = (FeatureObserver){0, 0, FLT_MIN, FLT_MAX};
                node->children[1]->class_observers[class].feature_observers[feature] = (FeatureObserver){0, 0, FLT_MIN, FLT_MAX};
            }
        }
    }

}

int predict(Node* tree, float* features){
    if (tree->is_leaf){
        float max_counter = 0;
        int max_index = 0;
        for (int i = 0; i < CLASS_NUMBER; i++){
            if (tree->class_observers[i].counter > max_counter){
                max_counter = tree->class_observers[i].counter;
                max_index = i;
            }
        }
        return max_index;
    }
    if (features[tree->feature] < tree->split_value){
        return predict(tree->children[0],features);
    } else {
        return predict(tree->children[1],features);
    }
}

void fit(Node* tree, float* features, int class_label){
    if (tree->is_leaf){
        if (class_label >= CLASS_NUMBER){
            //error
            return;
        }
        //update statistics
        update_statistics(tree, features, class_label);

        attemptToSplit(tree);
    } else {
        if (features[tree->feature] < tree->split_value){
            fit(tree->children[0],features,class_label);
        } else {
            fit(tree->children[1],features,class_label);
        }
    }
}

/*
 #include "HoeffdingTree.h"

void attemptToSplit(Node* node){
    float sum = 0;
    for (int i = 0; i < CLASS_NUMBER; i++){
        sum += node->counter[i];
    }
    if (sum < MIN_SAMPLES){
        return;
    }
    float best_gini = FLT_MIN;
    float second_best_gini = FLT_MIN;
    int best_feature = -1;
    float best_split_value = 0;
    //find the best split
    //add code to find the best split

    //check the Hoeffding bound
    float R = 1.0;
    float delta = 0.5;
    float n = sum;
    float epsilon = sqrt((R*R*log(1.0/delta))/(2.0*n));
    if (best_gini - second_best_gini > epsilon){
        //splitting
        node->is_leaf = 0;
        node->feature = best_feature;
        node->split_value = best_split_value;
        node->children[0] = (Node*)malloc(sizeof(Node));
        node->children[1] = (Node*)malloc(sizeof(Node));
        for (int i = 0; i < CLASS_NUMBER; i++){
            node->children[0]->counter[i] = 0;
            node->children[1]->counter[i] = 0;
        }
    }

}

int predict(Node* tree, float* features){
    if (tree->is_leaf){
        float max_counter = 0;
        int max_index = 0;
        for (int i = 0; i < CLASS_NUMBER; i++){
            if (tree->counter[i] > max_counter){
                max_counter = tree->counter[i];
                max_index = i;
            }
        }
        return max_index;
    }
    if (features[tree->feature] < tree->split_value){
        return predict(tree->children[0],features);
    } else {
        return predict(tree->children[1],features);
    }
}

void fit(Node* tree, float* features, int class_label){
    if (tree->is_leaf){
        if (class_label >= CLASS_NUMBER){
            //error
            return;
        }
        tree->counter[class_label]++;
        attemptToSplit(tree);
    } else {
        if (features[tree->feature] < tree->split_value){
            fit(tree->children[0],features,class_label);
        } else {
            fit(tree->children[1],features,class_label);
        }
    }
}
 */