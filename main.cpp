#include <iostream>
#include "MaterialClassifier/material_classifier.h"
#include <vl/generic.h>

using namespace std;

int main()
{
    VL_PRINT("Hello world!");
    MaterialClassifier classifier;
    MaterialParam param;
    param.buildChromaDictionary = false;
    param.buildFilterBank = false;
    param.buildTextonDictionary = false;
    param.buildSIFTDictionary = false;
    param.buildSIFTGmmDist = false;

    param.useSIFT = false;
    param.useTexton = false;
    param.useChroma = false;
    param.useSiftIFV = true;

    param.computeEigen = false;
    param.useComputeFeatureSet = false;

//    classifier.buildFilterKernelSet("/home/shenyunjun/Data/FMD");
    classifier.train("/home/shenyunjun/Data/FMD", param);
//    classifier.train("/home/shenyunjun/Data/KTH_TIPS", param);
}

