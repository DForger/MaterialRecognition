#ifndef DATA_IO_H
#define DATA_IO_H
#include <opencv2/opencv.hpp>
#include <QtCore>
#include <QFile>
#include <QDir>
#include <iostream>
#include <string>
#include <fstream>
#include <Eigen/Dense>

#ifdef USE_PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#endif

namespace Miscellaneous{
namespace IO{


template<typename DataType>
void data2Text_(cv::Mat &data, std::string fileName){
    std::fstream fileOut;
    fileOut.open(fileName, std::ios::out);
    if(!fileOut.is_open()){
        return;
    }
    int nRow = data.rows;
    int nCol = data.cols;
    DataType *fpSrcPtr;
    for(int i = 0; i < nRow; ++i){
        fpSrcPtr = data.ptr<DataType>(i);
        for(int j = 0; j < (nCol-1); ++j){
            fileOut<<(*fpSrcPtr++)<<' ';
        }
        fileOut<<(*fpSrcPtr);
        if(i != (nRow-1)){
            fileOut<<std::endl;
        }
    }
}

template<typename DataType>
void data2Text_(std::vector<std::vector<DataType> > &data, std::string fileName){
    std::fstream fileOut;
    fileOut.open(fileName, std::ios::out);
    if(!fileOut.is_open()){
        return;
    }
    int nRow = data.size();
    int nCol = data[0].size();
    for(int i = 0; i < nRow; ++i){
        for(int j = 0; j < (nCol-1); ++j){
            fileOut<<data[i][j]<<'\t';
        }
        fileOut<<data[i][nCol-1];
        if(i != (nRow-1)){
            fileOut<<std::endl;
        }
    }
}

template<typename DataType>
void readText_(std::vector<std::vector<DataType> > &data, std::string filename){
    data.clear();
    {
        std::fstream fileReader;
        fileReader.open(filename,std::ios::in);

        if(!fileReader.is_open()){
            return;
        }
        std::string sLineBuffer;
        while(std::getline(fileReader, sLineBuffer)){
            std::istringstream buffer(sLineBuffer);
            float fTmp;
            std::vector<DataType> line;
            while(buffer>>fTmp){
                line.push_back(fTmp);
            }

            data.push_back(line);
        }
    }
}

template<typename DataType>
void readData_(cv::Mat &data, std::string filename){
    std::vector<std::vector<DataType> > vecDataSets;
    {
        std::fstream fileReader;
        fileReader.open(filename,std::ios::in);

        if(!fileReader.is_open()){
            return;
        }
        std::string sLineBuffer;
        while(std::getline(fileReader, sLineBuffer)){
            std::istringstream buffer(sLineBuffer);
            float fTmp = 1;
            std::vector<DataType> line;
            while(buffer>>fTmp){
                line.push_back(fTmp);
            }

            vecDataSets.push_back(line);
        }
    }
    int nCol = vecDataSets[0].size();
    int nRow = vecDataSets.size();
    data = cv::Mat_<DataType>(nRow, nCol);

    DataType *dstPtr;
    for(int i = 0; i < nRow; ++i){
        dstPtr = data.ptr<DataType>(i);
        for(int j = 0; j < nCol; ++j){
            *dstPtr++ = vecDataSets[i][j];
        }
    }
}

//template<typename DataType>
//void data2Text_(cv::Mat &data, QString fileName){
//    QFile fileOut(fileName);
//    if(!fileOut.open(QFile::WriteOnly|QFile::Truncate)){
//        return;
//    }
//    QTextStream out(&fileOut);
//    int n_featureLength = data.cols;
//    int n_dataSize = data.rows;
//    DataType *dataPtr;
//    int channel = data.channels();
//    int rowStep = channel*n_featureLength;
//    for(int i = 0; i < n_dataSize; ++i){
//        dataPtr = data.ptr<DataType>(i);
//        for(int j = 0; j < n_featureLength; ++j){
//            for(int k = 0; k < channel; ++k){
//                out<<*dataPtr;
//                if((j*channel+k)<(rowStep-1)){
//                    out<<'\t';
//                }
//                dataPtr++;
//            }
//        }
//        out<<'\n';
//    }
//}


//    void data2Text(Eigen::MatrixXf &data, QString fileName){
//        QFile fileOut(fileName);
//        if(!fileOut.open(QFile::WriteOnly|QFile::Truncate)){
//            return;
//        }
//        QTextStream out(&fileOut);
//        int n_featureLength = data.cols();
//        int n_dataSize = data.rows();
//        for(int i = 0; i < n_dataSize; ++i){
//            for(int j = 0; j < n_featureLength; ++j){
//                out<<data(i,j)<<' ';
//            }
//            out<<'\n';
//        }
//    }

//    template<typename T>
//    void readData_(Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic> &data, QString dataAddress, bool extend){
//        QFile fileIn(dataAddress);
//        if(!fileIn.open(QIODevice::ReadOnly)){
//            return;
//        }

//        //QDataStream In(&fileIn);
//        QTextStream txtIn(&fileIn);

//        //get feature dimension
//        QString tmp = txtIn.readLine();
//        QStringList dataList = tmp.split(QRegExp("\\s|\\t|\\n"));
//        for(int i = dataList.size()-1; i >= 0; --i){
//            if(dataList[i] == '\0'){
//                dataList.removeAt(i);
//            }
//        }
//        int featureDim = dataList.size();
//        QString tmpData = txtIn.readAll();
//        dataList += tmpData.split(QRegExp("\\s|\\t|\\n"));
//        for(int i = dataList.size()-1; i >= 0; --i){
//            if(dataList[i] == '\0'){
//                dataList.removeAt(i);
//            }
//        }


//        int dataRows = dataList.size()/(featureDim);

//        //construct data
//        if(extend){
//            data = Eigen::Matrix<T, dataRows, featureDim+1>::Ones();
//        }else{
//            data = Eigen::Matrix<T, dataRows, featureDim>;
//        }

//        for(int i = 0; i < dataRows; ++i){
//            int beginPos = i*featureDim;
//            for(int j = 0; j < featureDim; ++j){
//                data(i,j) = static_cast<T>(dataList[beginPos+j].toDouble());
//            }
//        }
//    }

template<typename T>
void readData_(cv::Mat &data, QString dataAddress, bool extend){
    QFile fileIn(dataAddress);
    if(!fileIn.open(QIODevice::ReadOnly)){
        return;
    }

    //QDataStream In(&fileIn);
    QTextStream txtIn(&fileIn);

    //get feature dimension
    QString tmp = txtIn.readLine();
    QStringList dataList = tmp.split(QRegExp("\\s|\\t|\\n"));
    for(int i = dataList.size()-1; i >= 0; --i){
        if(dataList[i] == '\0'){
            dataList.removeAt(i);
        }
    }
    if(dataList.size() == 0){
        return;
    }
    int featureDim = dataList.size();
    QString tmpData = txtIn.readAll();
    dataList += tmpData.split(QRegExp("\\s|\\t|\\n"));
    for(int i = dataList.size()-1; i >= 0; --i){
        if(dataList[i] == '\0'){
            dataList.removeAt(i);
        }
    }


    int dataRows = dataList.size()/(featureDim);

    //construct data
    if(extend){
        data = cv::Mat_<T>::ones(dataRows, featureDim+1);
    }else{
        data = cv::Mat_<T>(dataRows, featureDim);
    }

    T *dataPtr;
    for(int i = 0; i < dataRows; ++i){
        int beginPos = i*featureDim;
        dataPtr = data.ptr<T>(i);
        for(int j = 0; j < featureDim; ++j){
            (*dataPtr++) = static_cast<T>(dataList[beginPos+j].toDouble());
        }
    }
}

template<typename T>
void readData_(cv::Mat &data, cv::Mat &label, QString dataAddress, bool extend = false){

    QFile fileIn(dataAddress);
    if(!fileIn.open(QIODevice::ReadOnly)){
        return;
    }

    //    QDataStream In(&fileIn);
    QTextStream txtIn(&fileIn);

    //get feature dimension and store data in
    QString tmp = txtIn.readLine();
    QStringList dataList = tmp.split(QRegExp("\\s|\\t|\\n"));
    for(int i = dataList.size()-1; i >= 0; --i){
        if(dataList[i] == '\0'){
            dataList.removeAt(i);
        }
    }

    int featureDim = dataList.size()-1;//remove the label
    QString tmpData = txtIn.readAll();
    dataList += tmpData.split(QRegExp("\\s|\\t|\\n"));
    for(int i = dataList.size()-1; i >= 0; --i){
        if(dataList[i] == '\0'){
            dataList.removeAt(i);
        }
    }
    int dataRows = dataList.size()/(featureDim+1);

    //construct data
    label = cv::Mat_<T>(dataRows, 1);
    if(extend){
        data = cv::Mat_<T>::ones(dataRows, featureDim+1);
    }else{
        data = cv::Mat_<T>(dataRows, featureDim);
    }

    T *dataPtr;
    for(int i = 0; i < dataRows; ++i){
        int beginPos = i*(featureDim+1);
        dataPtr = data.ptr<T>(i);
        for(int j = 0; j < featureDim; ++j){
            (*dataPtr++) = static_cast<T>(dataList[beginPos+j].toDouble());
        }
        label.at<T>(i,0) = static_cast<T>(dataList[beginPos+featureDim].toDouble());
    }

#ifdef DEBUG_DATA_READ
    int nCol = data.cols;
    int nRow = data.rows;
    for(int i = 0; i < nRow; ++i){
        std::cout<<i+1<<"th line ";
        T *ptr = data.ptr<T>(i);
        for(int j = 0; j < nCol; ++j){
            std::cout<<(*ptr++)<<" ";
        }
        std::cout<<std::endl;
    }
#endif

}

#ifdef USE_PCL
template<typename PointType>
void savePCD_(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, std::string fileName){
    pcl::io::savePCDFile(fileName, *cloud);
}
#endif

}
}
#endif // DATA_IO_H
