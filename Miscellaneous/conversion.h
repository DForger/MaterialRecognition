#ifndef CONVERSION_H
#define CONVERSION_H
#include <opencv2/opencv.hpp>
//#include <armadillo>

#ifdef USE_PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#endif

#include "Device/device.h"
#include "StereoMatch/stereo_base.hpp"

using namespace Stereo;

namespace Miscellaneous{
namespace Conversion{

const float maxDepth = 300000;

template<typename DispType, class PointType>
void disp2PointCloud_(cv::Mat dispImg,
                      cv::Mat rgbImg,
                      const StereoDevice &device,
                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                      bool isDense = true){
    if(dispImg.empty()){
        return;
    }

    int nCol = dispImg.cols;
    int nRow = dispImg.rows;

    float fDispDepthProduct = device.DispDepthProduct;

    Eigen::Matrix3f cameraKK = device.rightKK_rect;
    Eigen::Matrix3f cameraInvKK = cameraKK.inverse();
    cloud->clear();

    //        cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    int nDensePointCnt = 0;
    uchar *rgbPtr;
    DispType *dispPtr;
    for(int i = 0; i < nRow; ++i){
        rgbPtr = rgbImg.ptr<uchar>(i);
        dispPtr = dispImg.ptr<DispType>(i);
        for(int j = 0; j < nCol; ++j){
            PointType tmp;
            tmp.b = (*rgbPtr++);
            tmp.g = (*rgbPtr++);
            tmp.r = (*rgbPtr++);

            float fDisp = *dispPtr++;

            if(fDisp < 2.0f){
                if(isDense){
                    continue;
                }else{
                    tmp.x = NAN;
                    tmp.y = NAN;
                    tmp.z = NAN;
                }
            }else{
                float fDepth = fDispDepthProduct/fDisp;
                Vector3f pixel;
                pixel[0] = j;
                pixel[1] = i;
                pixel[2] = 1;
                Vector3f w = cameraInvKK*pixel;
                w = w*fDepth/w[2];
                tmp.x = w[0];
                tmp.y = w[1];
                tmp.z = w[2];

            }
            ++nDensePointCnt;
            cloud->push_back(tmp);
        }
    }

    if(isDense){
        cloud->width = nDensePointCnt;
        cloud->height = 1;
        cloud->is_dense = true;
    }else{
        cloud->width = nCol;
        cloud->height = nRow;
        cloud->is_dense = false;
    }
}


    template<typename DepthType>
    void depth2PointCloud_  (cv::Mat &depthImg,
                            cv::Mat &rgbImg,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                            const StereoDevice &device,
                             DepthType maxDepth = 0){
        MatrixXf baseKK_new = device.rightKK_rect;
        MatrixXf baseKKInv = baseKK_new.inverse();

        int nCol = depthImg.cols;
        int nRow = depthImg.rows;

        cloud->width = nCol;
        cloud->height = nRow;
        cloud->is_dense = false;

        if(maxDepth == 0){
            maxDepth = std::numeric_limits<DepthType>::max();
        }


        uchar *rgbPtr;
        DepthType *depthPtr;
        for(int i = 0; i < nRow; ++i){
            rgbPtr = rgbImg.ptr<uchar>(i);
            depthPtr = depthImg.ptr<DepthType>(i);
            for(int j = 0; j < nCol; ++j){
                pcl::PointXYZRGB tmp;
                tmp.b = (*rgbPtr++);
                tmp.g = (*rgbPtr++);
                tmp.r = (*rgbPtr++);

                double depth = *depthPtr++;
                if((depth == 0)||(depth > maxDepth)){
                    tmp.x = NAN;
                    tmp.y = NAN;
                    tmp.z = NAN;
                }else{
                    Vector3f pixel;
                    pixel[0] = j;
                    pixel[1] = i;
                    pixel[2] = 1;
                    Vector3f w = baseKKInv*pixel;
                    w = w*depth/(w[2]);
                    tmp.x = w[0];
                    tmp.y = w[1];
                    tmp.z = w[2];
                }
                cloud->push_back(tmp);
            }
        }
    }

template<typename InputType, typename OutputType>
void discretizeDispImg(cv::Mat orgDispImg, cv::Mat &discDispImg, InputType step = 0.5){

   if(orgDispImg.empty()||step == 0){
       return;
   }
   int nCol = orgDispImg.cols;
   int nRow = orgDispImg.rows;

   discDispImg = cv::Mat_<OutputType>(nRow, nCol, (OutputType)0);

   InputType *srcPtr;
   OutputType *dstPtr;
   for(int i = 0; i < nRow; ++i){
       srcPtr = orgDispImg.ptr<InputType>(i);
       dstPtr = discDispImg.ptr<OutputType>(i);
       for(int j = 0; j < nCol; ++j){
           *dstPtr = (*srcPtr)/step;
           ++dstPtr;
           ++srcPtr;
       }
   }
}

//    template<typename DepthType>
//    void depth2PointCloud_(cv::Mat &depthImg,
//                          cv::Mat &rgbImg,
//                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud){
//        MatrixXf baseKK_new = MatrixXf(3,3);
////        baseKK_new << 264,  0,      157.953622490334,
////                      0,    264,	117.330259425079,
////                      0,    0,      1;
//        baseKK_new << 296.7737558758737, 0, 158.2762565612793,
//                    0, 296.7737558758737, 117.6855344772339,
//                    0, 0, 1;
//        MatrixXf baseKKInv = baseKK_new.inverse();

//        int nCol = depthImg.cols;
//        int nRow = depthImg.rows;

//        float factor = 10000;

//        cloud->width = nCol;
//        cloud->height = nRow;
//        cloud->is_dense = false;

////        cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

//        uchar *rgbPtr;
//        DepthType *depthPtr;
//        for(int i = 0; i < nRow; ++i){
//            rgbPtr = rgbImg.ptr<uchar>(i);
//            depthPtr = depthImg.ptr<DepthType>(i);
//            for(int j = 0; j < nCol; ++j){
//                pcl::PointXYZRGB tmp;
//                tmp.b = (*rgbPtr++);
//                tmp.g = (*rgbPtr++);
//                tmp.r = (*rgbPtr++);

//                double depth = *depthPtr++;
//                if((depth == 0)||(depth == maxDepth)){
//                    tmp.x = NAN;
//                    tmp.y = NAN;
//                    tmp.z = NAN;
//                }else{
//                    Vector3f pixel;
//                    pixel[0] = j;
//                    pixel[1] = i;
//                    pixel[2] = 1;
//                    Vector3f w = baseKKInv*pixel;
//                    w = w*depth/(factor*w[2]);
//                    tmp.x = w[0];
//                    tmp.y = w[1];
//                    tmp.z = w[2];
//                }
//                cloud->push_back(tmp);
//            }
//        }
//    }

//template <typename ArmaDataType, typename CvDataType>
//void armamat2cvmat_(arma::Mat<ArmaDataType> &amat, cv::Mat &cvmat){
//    int nRows = amat.n_rows;
//    int nCols = amat.n_cols;

//    cvmat = cv::Mat_<CvDataType>(nRows, nCols);

//    CvDataType *cvmatPtr;
//    for(int i = 0; i < nRows; ++i){
//        cvmatPtr = cvmat.ptr<CvDataType>(i);
//        for(int j = 0; j < nCols; ++j){
//            *cvmatPtr++ = amat(i,j);
//        }
//    }
//}

//template <typename CvDataType, typename ArmaDataType>
//void cvmat2armamat_(cv::Mat &cvmat, arma::Mat<ArmaDataType> &amat){
//    int nRow = cvmat.rows;
//    int nCol = cvmat.cols;
//    amat.set_size(nRow, nCol);

//    CvDataType *cvmatPtr;
//    for(int i = 0; i < nRow; ++i){
//        cvmatPtr = cvmat.ptr<CvDataType>(i);
//        for(int j = 0; j < nCol; ++j){
//            amat(i,j) = *cvmatPtr++;
//        }
//    }
//}


template <typename DispType, typename DepthType>
void disp2depth_(cv::Mat &dispImg, cv::Mat &depthImg, StereoParam param)
{
    int nCol = dispImg.cols;
    int nRow = dispImg.rows;
    depthImg = cv::Mat_<DepthType>(nRow, nCol);

    double factor = param.f_scaledDispDepthProduct;

    DepthType *depthPtr;
    DispType *dispPtr;
    for(int i = 0; i < nRow; ++i){
        dispPtr = dispImg.ptr<DispType>(i);
        depthPtr = depthImg.ptr<DepthType>(i);
        DepthType depth;
        for(int j = 0; j < nCol; ++j){
            float disp = *dispPtr++;
            if(disp < 0.1){
                depth = 0;
            }else{
                depth = factor/disp;
            }
            *depthPtr++ = depth;
        }
    }
}

//    void depth2Gray(cv::Mat &depthImg, cv::Mat &grayImg){
//        cv::Mat _grayImg;
//        _grayImg.create(depthImg.rows,depthImg.cols,CV_8UC1);

//        ushort *src;
//        uchar *tgt;
//        int Cols = _grayImg.cols;
//        int Rows = _grayImg.rows;
//        for(int i = 0; i < Rows; ++i){
//            src = depthImg.ptr<ushort>(i);
//            tgt = _grayImg.ptr<uchar>(i);
//            for(int j = 0; j < Cols; ++j){
//                *tgt = static_cast<uchar>((*src)*255/std::numeric_limits<ushort>::max());
//                tgt++;
//                src++;
//            }
//        }

//        grayImg = _grayImg;
//    }

template<typename DepthType, typename DispType>
void depth2horDisp_(cv::Mat &depthImg,
                    cv::Mat &dispImg,
                    StereoParam param = StereoParam(),
                    int depthUnit = DEPTH_UNIT_100UM)
{
    int n_col = depthImg.cols;
    int n_row = depthImg.rows;

    float fScale  = 1;
    switch(depthUnit){
    case DEPTH_UNIT_100UM:
        fScale = 10000;
        break;
    case DEPTH_UNIT_1M:
        fScale = 1;
        break;
    case DEPTH_UNIT_1MM:
        fScale = 1000;
        break;
    default:
        break;
    }

    float fFactor = param.f_scaledDispDepthProduct*fScale;
    cv::Mat disparity = cv::Mat(n_row, n_col,CV_32F);
    DepthType *pDepthPtr;
    DispType *pDisPtr;
    for(int i = 0; i < n_row; i++){
        pDepthPtr = depthImg.ptr<DepthType>(i);
        pDisPtr = disparity.ptr<DispType>(i);
        for(int j = 0; j < n_col; j++){
            float depth = *pDepthPtr++;
            if(depth <= 0){
                *pDisPtr++ = 0;
                continue;
            }
            //            Vector3f rgbPixel, irPixel, w_ir, w_rgb;
            //            rgbPixel[0] = j;
            //            rgbPixel[1] = i;
            //            rgbPixel[2] = 1;


            //            depth = depth/n_depth2MeterFactor;
            //            w_rgb = baseKKInv_new*rgbPixel;
            //            double scale = depth/w_rgb[2];
            //            w_rgb = w_rgb*scale;
            //            w_ir = b2rRotation_new*w_rgb+b2rTranslation_new;

            //            irPixel = referKK_new*w_ir;
            //            irPixel[0] = irPixel[0]/irPixel[2];
            //            float dx = irPixel[0] - j;
            float dx = fFactor/depth;
            if(dx > 80.0f){
                *pDisPtr++ = 0;
            }else{
                *pDisPtr++ = dx;
            }
        }
    }
    dispImg = disparity;
}

//void disparity2PointCloud(cv::Mat disparity, cv::Mat rgbImg,
//                                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
//                                          MatrixXf extMat, MatrixXf baseIntMat, MatrixXf referIntMat){
//    if(disparity.empty()){
//        return;
//    }

//    int n_col = disparity.cols;
//    int n_row = disparity.rows;

//    cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

//    float *disPtr;
//    uchar *rgbPtr;
//    for(int i = 0; i < n_row; i++){
//        disPtr = disparity.ptr<float>(i);
//        rgbPtr = rgbImg.ptr<uchar>(i);
//        for(int j = 0; j < n_col; ++j){
//            float vx = disPtr[0];
//            float vy = disPtr[1];
//            if((vx == 0)&&(vy == 0)){
//                disPtr+=2;
//                rgbPtr+=3;
//                continue;
//            }
//            Vector3f b,r,w;
//            b[0] = j,b[1] = i, b[2] = 1;
//            r[0] = j+vx, r[1] = i+vy, r[2] = 1;
//            b = baseIntMat.inverse()*b;
//            r = referIntMat.inverse()*r;

//            MatrixXf A(4,3);
////            A(0,0)=extMat(2,0)*b(0)-extMat(0,0), A(0,1)=extMat(2,1)*b(0)-extMat(0,1), A(0,2)=extMat(2,2)*b(0)-extMat(0,2);
////            A(1,0)=extMat(2,0)*b(1)-extMat(1,0), A(1,1)=extMat(2,1)*b(1)-extMat(1,1), A(1,2)=extMat(2,2)*b(1)-extMat(1,2);
//            A(0,0)=-1,A(0,1)=0,A(0,2)=b(0);
//            A(1,0)=0,A(1,1)=-1,A(1,2)=b(1);
//            A(2,0)=extMat(2,0)*r(0)-extMat(0,0), A(2,1)=extMat(2,1)*r(0)-extMat(0,1), A(2,2)=extMat(2,2)*r(0)-extMat(0,2);
//            A(3,0)=extMat(2,0)*r(1)-extMat(1,0), A(3,1)=extMat(2,1)*r(1)-extMat(1,1), A(3,2)=extMat(2,2)*r(1)-extMat(1,2);

//            Vector4f B;
////            B(0)=extMat(0,3)-extMat(2,3)*b(0);
////            B(1)=extMat(1,3)-extMat(2,3)*b(1);
//            B(0)=0, B(1)=0, B(2)=extMat(0,3)-extMat(2,3)*r(0),B(3)=extMat(1,3)-extMat(2,3)*r(1);
//            w = (A.transpose()*A).inverse()*A.transpose()*B;
//            if(w[2] < 0){
//                w = -w;
//            }
//            if(w[2] > 20){
//                continue;
//                disPtr += 2;
//                rgbPtr += 3;
//            }

//            pcl::PointXYZRGB p;
//            p.x = w[0], p.y = w[1], p.z = w[2];
//            p.r = rgbPtr[2], p.g = rgbPtr[1], p.b = rgbPtr[0];
//            cloud->push_back(p);

//            disPtr += 2;
//            rgbPtr += 3;
//        }
//    }
//}

}
}

#endif // CONVERSION_H
