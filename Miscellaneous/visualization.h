#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <opencv2/opencv.hpp>

#ifdef USE_PCL
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#endif

namespace Miscellaneous{


template<typename DataType>
void plotCurve(cv::Mat &_curveData, cv::Mat &curveMap){
    cv::Mat curveData = _curveData.clone();
    int dataLength = curveData.cols;
    int nStep = 400/dataLength;

    double min, max;
    cv::minMaxLoc(curveData, &min , &max);
    curveData = curveData - min;
    float scale = 180/(max-min);
//    float scale = 180;
    curveMap = cv::Mat_<uchar>(200,400,(uchar)0);

    DataType *srcPtr = curveData.ptr<float>(0);
    DataType preData = *srcPtr;
    srcPtr++;
    float curData;
    for(int i = 1; i < dataLength; ++i){
        curData = *srcPtr;

        cv::Point start,end;
        start.x = (i-1)*nStep;
        end.x = i*nStep;

        start.y = 200 - preData*scale;
        end.y = 200 - curData*scale;

        cv::line(curveMap, start, end, cv::Scalar(255),2);

        srcPtr++;
        preData = curData;
    }
}

template<typename DataType>
void plotHist(cv::Mat &hist, cv::Mat &fea){
    int feaLength = fea.cols;
    int imgHeight = 400;
    int histVtScale = 80;
    int histHzScale = 10;
    int imgHeightHalf = imgHeight/2;
    hist = cv::Mat_<uchar>(imgHeight,feaLength*histHzScale, (uchar)0);
    for(int i = 0; i < feaLength; ++i){
        int startY,height;
        if(fea.at<DataType>(0,i) > 0){
            startY = -(int)fea.at<DataType>(0,i)*histVtScale+imgHeightHalf;
            if(startY < 0){
                startY = 0;
                height = imgHeightHalf;
            }else{
                height = fea.at<DataType>(0,i)*histVtScale;
            }
        }else{
            startY = imgHeightHalf;
            height = -fea.at<DataType>(0,i)*histVtScale;
            if(height > imgHeightHalf){
                height = imgHeightHalf;
            }
        }

        cv::rectangle(hist,cv::Rect(i*10,startY,5,height),cv::Scalar(255),2);
    }
}

#ifdef USE_PCL
template<class PointType>
void visualizePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud){
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D viewer"));
    viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(cloud);
    viewer->addPointCloud<PointType>(cloud, rgb, "cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->spin();
    viewer->close();
}

template<class PointType>
void visualizePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &orgCloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &postCloud){
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D viewer"));
    int orgViewPort = 0;
    int postViewPort = 1;
    viewer->setBackgroundColor(0,0,0);

    viewer->createViewPort(0, 0, 0.5, 1, orgViewPort);
    viewer->createViewPort(0.5, 0, 1, 1, postViewPort);
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> orgRgb(orgCloud);
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> postRgb(postCloud);
    viewer->addPointCloud<PointType>(orgCloud, orgRgb, "org",orgViewPort);
    viewer->addPointCloud<PointType>(postCloud, postRgb, "post", postViewPort);
//    viewer->addCoordinateSystem(1.0);
    viewer->spin();
    viewer->close();
}
#endif

}

#endif // VISUALIZATION_H
