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
#include <vl/gmm.h>
#include <vl/fisher.h>

struct MaterialFeatureSet{
    std::vector<cv::Mat> vecMaterialFeatures;
    cv::Mat textonHistogram;
    cv::Mat chromaHistogram;
};

class GMM{
public:
    GMM()
        :mean(NULL),
        covariance(NULL),
        prior(NULL){

    }

    ~GMM(){
        if(mean != NULL){
            delete[] mean;
        }

        if(covariance != NULL){
            delete[] covariance;
        }

        if(prior != NULL){
            delete[] prior;
        }
    }

    bool load(std::string fileName, int dimension){
        cv::Mat gmmInfo;
        Miscellaneous::IO::readData_<float>(gmmInfo, fileName);

        if(gmmInfo.cols != (dimension*2+1)){
            return false;
        }

        if(mean != NULL){
            delete[] mean;
        }
        if(covariance != NULL){
            delete[] covariance;
        }
        if(prior != NULL){
            delete[] prior;
        }

        this->clusterNum = gmmInfo.rows;
        this->dimension = dimension;
        mean = new float[dimension*clusterNum];
        covariance = new float[dimension*clusterNum];
        prior = new float[clusterNum];
        vecMean.resize(clusterNum);
        vecCovMat.resize(clusterNum);


        for(int i = 0; i < clusterNum; ++i){
            float *gmmPtr = gmmInfo.ptr<float>(i);

            for(int j = 0; j < dimension; ++j){
                mean[i*dimension+j] = *gmmPtr;
                ++gmmPtr;
            }
            cv::Mat meanMat = cv::Mat_<float>(1, dimension, mean+i*dimension);
            vecMean[i] = meanMat;

            cv::Mat covMat = cv::Mat_<float>(dimension, dimension, float(0));
            for(int j = 0; j < dimension; ++j){
                covariance[i*dimension+j] = *gmmPtr;
                covMat.at<float>(j,j) = *gmmPtr;
                ++gmmPtr;
            }
            vecCovMat[i] = covMat;

            prior[i] = *gmmPtr;
        }

        return true;
    }

    int dimension;
    int clusterNum;
    float *mean;
    float *covariance;
    float *prior;
    std::vector<cv::Mat> vecMean;
    std::vector<cv::Mat> vecCovMat;

};

struct MaterialParam{

    MaterialParam()
        : useTexton(true),
          useSIFT(true),
          useChroma(false),
          useSiftIFV(false),
          buildTextonDictionary(false),
          buildFilterBank(false),
          buildSIFTDictionary(false),
          buildSIFTGmmDist(false),
          buildChromaDictionary(false),
          useComputeFeatureSet(false),
          computeEigen(true){}

    bool useTexton;
    bool useSIFT;
    bool useChroma;
    bool useSiftIFV;

    bool buildTextonDictionary;
    bool buildFilterBank;
    bool buildSIFTDictionary;
    bool buildSIFTGmmDist;
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

    void extractSiftIFV(cv::Mat img, cv::Mat mask, cv::Mat &siftIfv, GMM &gmmDist);

    void buildFilterKernelSet(std::string dataAddress);

//    void buildTextonDictionarySet(std::string dataAddress);

    void buildGlobalTextonDictionary(std::string dataAddress);

    void buildSIFTDictionary(std::string dataAddress);

    void buildSiftIFVenCoder(std::string dataAddress);


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
    static size_t ms_nSiftIFVDimension;

    std::vector<size_t> m_vecClassModelSize;
    std::vector<cv::Mat> m_vecClassModelSet;
    std::vector<std::vector<MaterialFeatureSet> > m_vecClassMaterialModelSets;

    cv::Mat m_globalTextonDictionary;
    cv::Mat m_siftDictionary;
    GMM m_siftGMMDist;

    //data dir
    std::map<int, std::string> mapIndex2FileDirectory;



};

#endif // MATERIALCLASSIFIER_H
