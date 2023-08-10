#include <Arduino.h>
#include "XPowersLib.h"

#include <vector>
#include <string>
#include <sstream>

#include <naiveBayes.h>
#include <HoeffdingTree.h>


//#define DEBUG


unsigned int FEATURE_COUNT = 1;

unsigned long lastTime = 0;

int partitionDim = 1;


//Naive_Bayes classifier = Naive_Bayes(2);

HoeffdingTree classifier(0.0000001, 10);

//dataset definiton
std::vector<float> feature(FEATURE_COUNT,0.0);
int expectedLabel;
XPowersAXP2101 PMU;


void setup() {
  PMU.begin(Wire, AXP2101_SLAVE_ADDRESS, 38, 39);
  PMU.setDC4Voltage(1200);   // Here is the FPGA core voltage. Careful review of the manual is required before modification.
  PMU.setALDO1Voltage(3300); // BANK0 area voltage
  PMU.setALDO2Voltage(3300); // BANK1 area voltage
  PMU.setALDO3Voltage(2500); // BANK2 area voltage
  PMU.setALDO4Voltage(1800); // BANK3 area voltage

  PMU.enableALDO1();
  PMU.enableALDO2();
  PMU.enableALDO3();
  PMU.enableALDO4();

  Serial.begin(115200);
  Serial1.begin(500000, SERIAL_8N1, 4,5);
  delay(100);

  Serial.println("Serial1 started");

  Serial1.println("Serial started");
  Serial1.println("waiting for configuration msg");
  while (Serial1.available() == 0) {
    delay(100);
    Serial1.println("waiting for configuration msg");
  }
  std::string cfg = Serial1.readStringUntil('\n').c_str();
  while(cfg.find(',') != std::string::npos){
    //Serial1.println("chk1");
    std::string arg = cfg.substr(0,cfg.find(','));
    //Serial1.println(arg.c_str());
    if (arg.substr(0,arg.find(':')).find("partitionDim") != std::string::npos) {
      //Serial1.println(arg.substr(cfg.find(':')+1).c_str());
      partitionDim = atoi(arg.substr(cfg.find(':')+1).c_str());
    }
    if (arg.substr(0,arg.find(':')).find("featureCount") != std::string::npos) {
      //Serial1.println(arg.substr(cfg.find(':')+1).c_str());
      FEATURE_COUNT = atoi(arg.substr(cfg.find(':')+1).c_str());
      feature.resize(FEATURE_COUNT,0.0);
    }
    cfg.erase(0, cfg.find(',') + 1);
  }
  Serial1.println("configuration msg received");
  Serial1.println("partitionDim: " + String(partitionDim) + ", featureCount: " + String(FEATURE_COUNT));

  

  Serial1.println("start sending data");
}




void loop() {
  std::string toPrint = "";
  if (Serial1.available() > 0) {
    std::string data = Serial1.readStringUntil('\n').c_str();

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
        //Serial1.println(singleFeature.c_str() + String(" ") + String(i));
      }
      //Serial.println("m feature size: " + String(feature.size()));
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
  Serial1.println(toPrint.c_str());
  }
}



