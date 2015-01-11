#ifndef MATERIALCLASSIFIER_H
#define MATERIALCLASSIFIER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include "Preprocessing/SLICSuperpixel.h"
#include "Miscellaneous/miscellaneous.h"
#include <QDir>
#include <QtCore>
#include <opencv2/xfeatures2d.hpp>

#include <vl/generic.h>
#include <vl/dsift.h>

struct MaterialFeatureSet{
    std::vector<cv::Mat> vecMaterialFeatures;
    cv::Mat textonHistogram;
    cv::Mat chromaHistogram;
};

struct MaterialParam{

    MaterialParam()
        : useTexton(true),
          useSIFT(true),
          useChroma(false),
          buildTextonDictionary(false),
          buildFilterBank(false),
          buildSIFTDictionary(false),
          buildChromaDictionary(false),
          useComputeFeatureSet(false),
          computeEigen(true){}

    bool useTexton;
    bool useSIFT;
    bool useChroma;

    bool buildTextonDictionary;
    bool buildFilterBank;
    bool buildSIFTDictionary;
    bool buildChromaDictionary;

    bool useComputeFeatureSet;

    bool computeEigen;
};


class MaterialClassifier
{
public:
    MaterialClassifier();

    ~MaterialClassifier();

    void train(std::string dataAddress, MaterialParam &param);

    void load(std::string paraAddress);

    void test(cv::Mat img);

    //getter and setter

    void extractClusterFeature(cv::Mat img,
                               SLICSuperpixel &slic,
                               std::vector<MaterialFeatureSet> &clusterFeatureSet);

    void extractTextonDist(cv::Mat img, cv::Mat mask, cv::Mat &textonDist);

    void extractSIFTDist(cv::Mat img, cv::Mat mask, cv::Mat &siftDist);

    void buildFilterKernelSet(std::string dataAddress);

//    void buildTextonDictionarySet(std::string dataAddress);

    void buildGlobalTextonDictionary(std::string dataAddress);

    void buildSIFTDictionary(std::string dataAddress);

    void buildChromaDictionary(std::string dataAddress);

    void buildColorModelSet(std::string dataAddress);

    void buildModelSet(std::string dataAddress);

private:
    inline bool fileExistenceTest (const std::string& name) {
        if (FILE *file = fopen(name.c_str(), "r")) {
            fclose(file);
            return true;
        } else {
            return false;
        }
    }

private:
    size_t m_nFilterKernelWidth;    //kernel width NxN
    size_t m_nFilterKernelSetSize;  //how many kernel we use
    std::vector<cv::Mat> m_vecFilterKernelSet;

    static size_t ms_nGlobalTextonDictionarySize;
    static size_t ms_nSiftDictionarySize;
    static size_t ms_nChromaDictionarySize;

    std::vector<size_t> m_vecClassModelSize;
    std::vector<cv::Mat> m_vecClassModelSet;
    std::vector<std::vector<MaterialFeatureSet> > m_vecClassMaterialModelSets;

    cv::Mat m_globalTextonDictionary;
    cv::Mat m_siftDictionary;

};

#endif // MATERIALCLASSIFIER_H
