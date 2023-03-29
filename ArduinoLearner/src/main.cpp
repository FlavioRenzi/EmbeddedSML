#include <Arduino.h>
#include <vector>
#include <string>
#include <sstream>

#include <naiveBayes.h>

//#define DEBUG


#define FEATURE_COUNT 5

unsigned long lastTime = 0;

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
int expectedLabel;

void loop() {
  std::string toPrint = "";
  if (Serial.available() > 0) {
    std::string data = Serial.readStringUntil('\n').c_str();

    //parsing input string
    std::stringstream sstr(data);
    int x = 0;
    while(sstr.good() && x < partitionDim){
      x++;
      std::string substr;
      std::string singleFeature;
      getline(sstr, substr, ';');
      std::stringstream subsstr(substr);
      
      for (int i = 0; i < FEATURE_COUNT; i++) {
        getline(subsstr, singleFeature, ',');
        feature[i] = std::stof(singleFeature);
      }
      getline(subsstr, singleFeature, ',');
      expectedLabel = atoi(singleFeature.c_str());

      //prediction
      toPrint.append("predicted: ");
      lastTime = micros();
      toPrint.append(std::to_string(classifier.predict(feature)));
      toPrint = toPrint.append(", prediction_time: " + std::to_string(micros() - lastTime));

      //model update
      lastTime = micros();
      classifier.fit(feature, expectedLabel);
      toPrint = toPrint.append(", train_time: " + std::to_string(micros() - lastTime));

      toPrint.append(","); 
      toPrint.append(";");
    }
  Serial.println(toPrint.c_str());
  }
}



