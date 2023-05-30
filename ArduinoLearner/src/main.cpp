#include <Arduino.h>
#include <SoftwareSerial.h>

#include <vector>
#include <string>
#include <sstream>

#include <naiveBayes.h>
#include <HoeffdingTree.h>


//#define DEBUG


unsigned int FEATURE_COUNT = 1;

unsigned long lastTime = 0;

int partitionDim = 1;

SoftwareSerial Serialdbg;

Naive_Bayes classifier = Naive_Bayes(2);

//HoeffdingTree classifier(0.0000001, 10);

//dataset definiton
std::vector<float> feature(FEATURE_COUNT,0.0);
int expectedLabel;


void setup() {
  Serial.begin(500000);
  Serialdbg.begin(115200, SWSERIAL_8N1, 13, 15);
  delay(100);

  Serialdbg.println("Serialdbg started");

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
    if (arg.substr(0,arg.find(':')).find("featureCount") != std::string::npos) {
      //Serial.println(arg.substr(cfg.find(':')+1).c_str());
      FEATURE_COUNT = atoi(arg.substr(cfg.find(':')+1).c_str());
      feature.resize(FEATURE_COUNT,0.0);
    }
    cfg.erase(0, cfg.find(',') + 1);
  }
  Serial.println("configuration msg received");
  Serial.println("partitionDim: " + String(partitionDim) + ", featureCount: " + String(FEATURE_COUNT));

  

  Serial.println("start sending data");
}




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
      
      for (unsigned int i = 0; i < FEATURE_COUNT; i++) {
        getline(subsstr, singleFeature, ',');
        feature[i] = std::stof(singleFeature);
        //Serialdbg.println(singleFeature.c_str() + String(" ") + String(i));
      }
      //Serialdbg.println("m feature size: " + String(feature.size()));
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



