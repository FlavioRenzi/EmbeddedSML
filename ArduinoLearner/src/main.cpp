#include <Arduino.h>
#include <vector>
#include <string>
#include <sstream>

#include "naiveBayes.h"

//#define DEBUG


#define FEATURE_COUNT 5


int partitionDim = 1;


Naive_Bayes classifier(2);


void setup() {
  Serial.begin(500000);
  delay(100);

  Serial.println("Serial started");
  Serial.println("waiting for configuration msg");
  while (Serial.available() == 0) {
    delay(10);
  }
  std::string cfg = Serial.readStringUntil('\n').c_str();
  while(cfg.find(',') != std::string::npos){
    //Serial.println("chk1");
    std::string arg = cfg.substr(0,cfg.find(','));
    //Serial.println(arg.c_str());
    if (arg.substr(0,arg.find(':')).find("partitionDim") != std::string::npos) {
      //Serial.println(arg.substr(cfg.find(':')+1).c_str());
      partitionDim = atoi(arg.substr(cfg.find(':')+1).c_str());
    }
    cfg.erase(0, cfg.find(',') + 1);
  }
  Serial.println("configuration msg received");
  Serial.println("partitionDim: " + String(partitionDim));

  Serial.println("start sending data");
}


//dataset definiton
std::vector<float> feature(FEATURE_COUNT,0.0);
bool expectedLabel;

void loop() {
  std::string toPrint = "";
  if (Serial.available() > 0) {
    //Serial.println("data received");
    std::string data = Serial.readStringUntil('\n').c_str();
    //Serial.println(data.c_str());

    //parsing input string
    std::stringstream sstr(data);
    while(sstr.good()){
      std::string substr;
      std::string singleFeature;
      getline(sstr, substr, ';');
      std::stringstream subsstr(data);
      
      for (int i = 0; i < FEATURE_COUNT; i++) {
        getline(subsstr, singleFeature, ',');
        feature[i] = std::stof(singleFeature);
      }
      getline(subsstr, singleFeature, ',');
      expectedLabel = singleFeature.compare("b'UP'") == 0;


      std::vector<std::vector<float>> featureVector;
      featureVector.push_back(feature);
      toPrint = toPrint.append("predicted: ");
      toPrint = toPrint.append(std::to_string(classifier.predict(featureVector[0])));
      //Serial.print(toPrint.c_str());
      classifier.fit(featureVector, expectedLabel);
      toPrint = toPrint.append(","); 
    }
  Serial.println(toPrint.append("0;").c_str());
  }
}



