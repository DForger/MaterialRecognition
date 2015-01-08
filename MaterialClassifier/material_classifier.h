#ifndef MATERIALCLASSIFIER_H
#define MATERIALCLASSIFIER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include "Preprocessing/SLICSuperpixel.h"
#include "Miscellaneous/miscellaneous.h"

struct MaterialFeatureSet{
    std::vector<cv::Mat> vecMaterialFeatures;
    cv::Mat textonHistogram;
    cv::Mat chromaHistogram;
};


class MaterialClassifier
{
public:
    MaterialClassifier();

    ~MaterialClassifier();

    void train(std::string dataAddress);

    void load(std::string paraAddress);

    void test(cv::Mat img);

    //getter and setter

    void extractClusterFeature(cv::Mat img,
                               SLICSuperpixel &slic,
                               std::vector<MaterialFeatureSet> &clusterFeatureSet);

private:
    void buildFilterKernelSet(std::string dataAddress);

    void buildTextonDictionarySet(std::string dataAddress);

    void buildGlobalTextonDictionary(std::string dataAddress);

    void buildColorModelSet(std::string dataAddress);

    void buildModelSet(std::string dataAddress);

private:
    size_t m_nFilterKernelWidth;    //kernel width NxN
    size_t m_nFilterKernelSetSize;  //how many kernel we use
    std::vector<cv::Mat> m_vecFilterKernelSet;

    static size_t ms_nGlobalTextonDictionarySize;
    static size_t ms_nChromaHistSize;

    std::vector<size_t> m_vecClassModelSize;
    std::vector<cv::Mat> m_vecClassModelSet;
    std::vector<std::vector<MaterialFeatureSet> > m_vecClassMaterialModelSets;

    cv::Mat m_globalTextonDicitionary;

};

#endif // MATERIALCLASSIFIER_H
