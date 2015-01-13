#include <iostream>
#include "MaterialClassifier/material_classifier.h"
#include <vl/generic.h>

using namespace std;

int main()
{
    VL_PRINT("Hello world!");
    MaterialClassifier classifier;
    MaterialParam param;
    param.buildColorGmmDist = false;
    param.buildFilterBank = false;
    param.buildTextonDictionary = false;
    param.buildTextonGmmDist = false;
    param.buildSIFTDictionary = false;
    param.buildSIFTGmmDist = true;

    param.useSIFT = true;
    param.useTexton = false;
    param.useColorIFV= false;
    param.useSiftIFV = false;
    param.useTextonIFV = false;

    param.computeEigen = false;
    param.useComputeFeatureSet = true;


//    classifier.buildFilterKernelSet("/home/shenyunjun/Data/FMD");
    classifier.train("/home/shenyunjun/Data/FMD", param);
//    classifier.train("/home/shenyunjun/Data/new_data/fmd", param);
//    classifier.train("/home/shenyunjun/Data/KTH_TIPS", param);
}

