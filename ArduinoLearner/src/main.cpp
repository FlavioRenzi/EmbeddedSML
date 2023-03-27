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
  for (int partIndex = 0; partIndex < partitionDim; partIndex++){
    if (Serial.available() > 0) {
      std::string data = Serial.readStringUntil('\n').c_str();
      Serial.print(data.c_str());
      //parsing input string
      std::vector<std::string> v;
      std::stringstream sstr(data);
      while(sstr.good())
      {
          std::string substr;
          getline(sstr, substr, ',');
          v.push_back(substr);
      }
      for (int i = 0; i < FEATURE_COUNT; i++) {
        feature[i] = std::stof(v[i]);
        #ifdef DEBUG
        Serial.print(feature[i],6);
        Serial.print(",");
        #endif
      }
      expectedLabel = v[FEATURE_COUNT].compare("b'UP'") == 0;
      #ifdef DEBUG
      Serial.print(expectedLabel);
      #endif
      std::vector<std::vector<float>> featureVector;
      featureVector.push_back(feature);
      toPrint = toPrint.append("predicted: ");
      toPrint = toPrint.append(std::to_string(classifier.predict(featureVector[0])));
      
      classifier.fit(featureVector, expectedLabel);
      toPrint = toPrint.append(",");
    }
  }
  Serial.println(toPrint.append(";").c_str());
}



