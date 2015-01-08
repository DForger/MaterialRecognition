#ifndef MISCELLANEOUS_H
#define MISCELLANEOUS_H
//#define USE_PCL

#include <QtCore>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
//#include <armadillo>

#include "data_io.h"
//#include "conversion.h"
#include "visualization.h"
#include "interpolation.h"



#ifdef USE_PCL
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#endif


namespace Miscellaneous{

void generateMask(cv::Rect &baseRoi, cv::Rect &referRoi, cv::Mat depthImg,
                  float maxDisp, cv::Mat &baseMask, cv::Mat &referMask);

double disparityMapEvaluation(cv::Mat &calDispMap, cv::Mat &gtDispMap, cv::Mat &error_map);

void Erosion(cv::Mat &input, cv::Mat &output, int erosionType, int erosionSize);

void Dilation(cv::Mat &input, cv::Mat &output, int dilationType, int dilationSize);

void FindBlobPos(cv::Mat &input, std::vector<cv::Point> &posList);

//QStringList loadImageList(QString dataAddress);

template<typename DataType>
DataType getNZMeanValue(cv::Mat &data){
    if(data.empty()){
        return 0;
    }
    int nCol = data.cols;
    int nRow = data.rows;
    int nMinPixelNum = nCol*nRow*0.01;
    DataType *dataPtr;
    DataType meanValue = 0;
    int nCnt = 0;
    for(int i = 0; i < nRow; ++i){
        dataPtr = data.ptr<DataType>(i);
        for(int j = 0; j < nCol; ++j){
            if((*dataPtr) != (DataType)0){
                meanValue = meanValue + *dataPtr;
                ++nCnt;
            }
            dataPtr++;
        }
    }
    if(nCnt < nMinPixelNum){
        return 0;
    }else{
        return meanValue/nCnt;
    }
}

template<typename DepthTypeOrg, typename DepthTypeDst>
void mergeOrgDepth_(cv::Mat &orgDepth, cv::Mat &dstDepth){

    int n_row = orgDepth.rows;
    int n_col = orgDepth.cols;
    DepthTypeOrg *orgDepthPtr;
    DepthTypeDst *dstDepthPtr;
    for(int i = 0; i < n_row; ++i){
        orgDepthPtr = orgDepth.ptr<DepthTypeOrg>(i);
        dstDepthPtr = dstDepth.ptr<DepthTypeDst>(i);
        for(int j = 0; j < n_col; ++j){
            DepthTypeOrg depth = *orgDepthPtr++;
            if((depth == 0)||(depth >= 65535)){
                dstDepthPtr++;
            }else{
                *dstDepthPtr=depth;
                dstDepthPtr++;
            }
        }
    }
}

template<class T>
void MirrorFeaImg_(cv::Mat &orgFeaImg, cv::Mat &_mirrorFeaImg, size_t feaLength)
{
    int nStep = orgFeaImg.cols;
    int nCol = nStep/feaLength;
    int nRow = orgFeaImg.rows;
    cv::Mat mirrorFeaImg = cv::Mat_<T>(nRow, nStep);

    T *orgPtr, *mirrorPtr;
    for(int i = 0; i < nRow; ++i){

        for(int j = 0, k = nCol-1; j < nCol; j++,k--){
            orgPtr = orgFeaImg.ptr<T>(i)+feaLength*j;
            mirrorPtr = mirrorFeaImg.ptr<T>(i)+feaLength*k;
            for(int l = 0; l < feaLength; ++l){
                mirrorPtr[l] = orgPtr[l];
            }
        }
    }
    _mirrorFeaImg = mirrorFeaImg;
}

inline void normlizeArray(float *array, const int &length){
    float norm = 0;
    for(int i = 0; i < length; ++i){
        float value = array[i];
        norm = norm + value*value;
    }
    norm = std::sqrt(norm);
    for(int i = 0; i < length; ++i){
        array[i] = array[i]/norm;
    }
}

template<typename T>
void ErodeDispMap(cv::Mat &input, cv::Mat &output, int size = 1)
{
    if(input.empty()){
        return;
    }

    int nCol = input.cols;
    int nRow = input.rows;

    cv::Mat mask = cv::Mat_<uchar>(nRow, nCol, (uchar)0);
    {
        T *srcPtr;
        uchar *dstPtr;
        for(int i = 0; i < nRow; ++i){
            srcPtr = input.ptr<T>(i);
            dstPtr = mask.ptr<uchar>(i);
            for(int j = 0; j < nCol; ++j){
                if(*srcPtr !=  0){
                    *dstPtr = 1;
                }
                srcPtr++;
                dstPtr++;
            }
        }
    }

    Miscellaneous::Erosion(mask, mask, cv::MORPH_ELLIPSE, size);
    cv::Mat erodedMap = cv::Mat_<T>(nRow, nCol, (T)0);
    {
        uchar *maskPtr;
        T *srcPtr;
        T *dstPtr;
        for(int i = 0; i < nRow; ++i){
            srcPtr = input.ptr<T>(i);
            dstPtr = erodedMap.ptr<T>(i);
            maskPtr = mask.ptr<uchar>(i);
            for(int j = 0; j < nCol; ++j){
                if(*maskPtr != 0){
                    *dstPtr = *srcPtr;
                }
                maskPtr++;
                dstPtr++;
                srcPtr++;
            }
        }
    }

    output = erodedMap;
}

}

#endif // MISCELLANEOUS_H
