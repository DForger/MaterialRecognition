#include "miscellaneous.h"

namespace Miscellaneous {

//void Miscellaneous::Data2Text(cv::Mat &data, cv::Mat &label, QString fileName){
//    QFile fileOut(fileName);
//    if(!fileOut.open(QFile::WriteOnly|QFile::Truncate)){
//        return;
//    }
//    QTextStream out(&fileOut);
//    int n_featureLength = data.cols;
//    int n_dataSize = data.rows;
//    float *dataPtr;
//    for(int i = 0; i < n_dataSize; ++i){
//        dataPtr = data.ptr<float>(i);
//        for(int j = 0; j < n_featureLength; ++j){
//            out<<*dataPtr<<' ';
//            dataPtr++;
//        }
//        out<<label.at<float>(i);
//        out<<'\n';
//    }
//}

//template<typename T>
//void Miscellaneous::Data2Text_(cv::Mat &data, QString fileName,T num){

//}


//template<typename T>
//void Miscellaneous::readData_(cv::Mat &data, QString dataAddress, bool extend){
//    QFile fileIn(dataAddress);
//    if(!fileIn.open(QIODevice::ReadOnly)){
//        return;
//    }

//    //QDataStream In(&fileIn);
//    QTextStream txtIn(&fileIn);

//    //get feature dimension
//    QString tmp = txtIn.readLine();
//    QStringList dataList = tmp.split(QRegExp("\\s|\\t|\\n"));
//    for(int i = dataList.size()-1; i >= 0; --i){
//        if(dataList[i] == '\0'){
//            dataList.removeAt(i);
//        }
//    }
//    int featureDim = dataList.size();
//    QString tmpData = txtIn.readAll();
//    dataList += tmpData.split(QRegExp("\\s|\\t|\\n"));
//    for(int i = dataList.size()-1; i >= 0; --i){
//        if(dataList[i] == '\0'){
//            dataList.removeAt(i);
//        }
//    }


//    int dataRows = dataList.size()/(featureDim);

//    //construct data
//    if(extend){
//        data = cv::Mat_<T>::ones(dataRows, featureDim+1);
//    }else{
//        data = cv::Mat_<T>(dataRows, featureDim);
//    }

//    T *dataPtr;
//    for(int i = 0; i < dataRows; ++i){
//        int beginPos = i*featureDim;
//        dataPtr = data.ptr<T>(i);
//        for(int j = 0; j < featureDim; ++j){
//            (*dataPtr++) = static_cast<T>(dataList[beginPos+j].toDouble());
//        }
//    }
//}

//QStringList loadImageList(QString dataAddress){
//    QString fileName = dataAddress;
//    QFile fileIn(fileName);
//    if(!fileIn.open(QFile::ReadOnly)){
//        return QStringList();
//    }
//    QTextStream in(&fileIn);

//    QString imgDir = in.readAll();
//    QStringList imgDirList = imgDir.split(QRegExp("\\n"));

//    for(int i = 0; i < imgDirList.size();){
//        if(imgDirList[i] == ""){
//            imgDirList.removeAt(i);
//            continue;
//        }
//        ++i;
//    }
//    return imgDirList;
//}



void Erosion(cv::Mat &input, cv::Mat &output, int erosionType, int erosionSize){
    cv::Mat _output;
    cv::Mat element = cv::getStructuringElement(erosionType, cv::Size(2*erosionSize+1,2*erosionSize+1),
                                                cv::Point(erosionSize, erosionSize));
    cv::erode(input,_output,element);
    output = _output;
}

void Dilation(cv::Mat &input, cv::Mat &output, int dilationType, int dilationSize){
    cv::Mat _output;
    cv::Mat element = cv::getStructuringElement(dilationType, cv::Size(2*dilationSize+1,2*dilationSize+1),
                                                cv::Point(dilationSize, dilationSize));
    cv::dilate(input,_output,element);
    output = _output;

}

void FindBlobPos(cv::Mat &_input, std::vector<cv::Point> &posList){
    if(_input.empty()){
        return;
    }
    if(posList.size() != 0){
        posList.clear();
    }
    cv::Mat input = _input.clone();

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(input, contours,hierarchy,cv::RETR_CCOMP,cv::CHAIN_APPROX_NONE);

    //filter data
    int n_areaThresh = 50;
    double f_circleRatioThresh = 1.75;
    double f_widthHeightRatioThresh = 1.5;

    std::vector<cv::Point> centerPos;

    for(int i = contours.size()-1; i >=0; --i){
        if(hierarchy[i][3]!=-1){
            contours.erase(contours.begin()+i);
            continue;
        }
        double a = cv::contourArea(contours[i]);
        if(a < n_areaThresh){
            contours.erase(contours.begin()+i);
            continue;
        }

        double c = cv::arcLength(contours[i],true);
        if((c*c)/(4*3.1415926*a) > f_circleRatioThresh){
            contours.erase(contours.begin()+i);
            continue;
        }

        cv::Rect r= cv::boundingRect(cv::Mat(contours[i]));
        double width, height;
        width = r.tl().x-r.br().x;
        height = r.tl().y - r.br().y;
        if(width > height){
            double tmp = width;
            width = height;
            height = tmp;
        }
        if(height/width > f_widthHeightRatioThresh){
            contours.erase(contours.begin()+i);
        }
        cv::Point p;
        p.x = 0.5*(r.br().x+r.tl().x);
        p.y = 0.5*(r.br().y+r.tl().y);
        centerPos.push_back(p);
    }

    posList = centerPos;

}

//template<class T>
//void Miscellaneous::MirrorFeaImg_(cv::Mat &orgFeaImg, cv::Mat &_mirrorFeaImg, size_t feaLength)
//{
//    int nStep = orgFeaImg.cols;
//    int nCol = nStep/feaLength;
//    int nRow = orgFeaImg.rows;
//    cv::Mat mirrorFeaImg = cv::Mat_<T>(nRow, nStep);

//    float *orgPtr, *mirrorPtr;
//    for(int i = 0; i < nRow; ++i){

//        for(int j = 0, k = nCol-1; j < nCol; j++,k--){
//            orgPtr = orgFeaImg.ptr<T>(i)+feaLength*j;
//            mirrorPtr = mirrorFeaImg.ptr<T>(i)+feaLength*k;
//            for(int l = 0; l < feaLength; ++l){
//                mirrorPtr[l] = orgPtr[l];
//            }
//        }
//    }
//    _mirrorFeaImg = mirrorFeaImg;
//}


void generateMask(cv::Rect &baseRoi, cv::Rect &referRoi, cv::Mat _depthImg,
                                 float maxDisp, cv::Mat &_baseMask, cv::Mat &_referMask)
{
    cv::Mat depthImg;
    if(!_depthImg.empty()){
        Miscellaneous::ErodeDispMap<ushort>(_depthImg, depthImg);
    }

    int nRow = depthImg.rows;
    int nCol = depthImg.cols;
    cv::Mat baseMask = cv::Mat_<uchar>(nRow, nCol,(uchar)(0));
    cv::Mat referMask = cv::Mat_<uchar>(nRow, nCol, (uchar)(0));
    {
        cv::Point lt, rb;
        lt.x = std::max(baseRoi.x, referRoi.x);
        lt.y = std::max(baseRoi.y, referRoi.y);
        rb.x = std::min(baseRoi.x + baseRoi.width, referRoi.x + referRoi.width);
        rb.y = std::min(baseRoi.y + baseRoi.height, referRoi.y + referRoi.height);

        baseMask.adjustROI(-lt.y, rb.y-nRow, -lt.x, rb.x-nCol-maxDisp);
        referMask.adjustROI(-lt.y, rb.y-nRow, -lt.x, rb.x-nCol);
        baseMask = 1;
        referMask = 1;
        baseMask.adjustROI(lt.y, -rb.y+nRow, lt.x, -rb.x+nCol+maxDisp);
        referMask.adjustROI(lt.y, -rb.y+nRow, lt.x, -rb.x+nCol);
    }
//    if(!depthImg.empty()){
//        ushort *depthPtr;
//        for(size_t i = 0; i < nRow; ++i){
//            depthPtr = depthImg.ptr<ushort>(i);
//            for(size_t j = 0; j < nCol; ++j){
//                if(*depthPtr > 0){
//                    baseMask.at<uchar>(i,j) = 0;
//                    referMask.at<uchar>(i,j) = 0;
//                }
//                depthPtr++;
//            }
//        }
//    }

    _baseMask = baseMask;
    _referMask = referMask;
}


double  disparityMapEvaluation(cv::Mat &_calDispMap, cv::Mat &_gtDispMap, cv::Mat &error_map)
{
    if(_calDispMap.empty()||_gtDispMap.empty()){
        return 0;
    }
    cv::Mat calDispMap, gtDispMap;
    gtDispMap = _gtDispMap;
    int nRow = gtDispMap.rows;
    int nCol = gtDispMap.cols;
    if((_calDispMap.cols != nCol)||(_calDispMap.rows != nRow)){
        std::cout<<"resize disp map\n";
        std::cout.flush();
        float scale = _calDispMap.cols*1.0/nCol;
        cv::resize(_calDispMap, calDispMap, cv::Size(nCol, nRow),0, 0, cv::INTER_NEAREST);
        calDispMap = calDispMap/scale;
    }else{
        calDispMap = _calDispMap;
    }


    cv::Mat errorMap = cv::Mat_<cv::Vec3b>(nRow, nCol, cv::Vec3b(255,255,255));
    double d_error = 0;
    int n_validPixelCnt = 0;
    int n_invalidPixelCnt = 0;
    {
        float *calDispPtr, *gtDispPtr;
        for(int i = 0; i < nRow; ++i){
            calDispPtr = calDispMap.ptr<float>(i);
            gtDispPtr = gtDispMap.ptr<float>(i);
            for(int j = 0; j < nCol; ++j){
                if(*calDispPtr <= 0){
                    n_invalidPixelCnt++;
                }else{
                    if(*gtDispPtr != 0){
                        float diff = *gtDispPtr - *calDispPtr;
                        if(diff > 0){
                            uchar shift = static_cast<uchar>(diff*100);
                            cv::Vec3b p(255, 255-shift, 255-shift);
                            errorMap.at<cv::Vec3b>(i,j)= p;
                        }else{
                            diff = -diff;
                            uchar shift = static_cast<uchar>(diff*100);
                            cv::Vec3b p(255-shift, 255-shift, 255);
                            errorMap.at<cv::Vec3b>(i,j)= p;
                        }
                        d_error += diff;
                        n_validPixelCnt++;
                    }
                }
                calDispPtr++;
                gtDispPtr++;
            }
        }
    }
    std::cout<<"totoal error: "<<d_error<<std::endl;
    std::cout<<"counted pixel number: "<<n_validPixelCnt<<std::endl;
    std::cout<<"invalidPixel number: "<<n_invalidPixelCnt<<std::endl;
    std::cout<<"average disparity error: "<<d_error/n_validPixelCnt<<std::endl;
    cv::imshow("errorMap", errorMap);
    std::cout.flush();
    error_map = errorMap;
    return d_error/n_validPixelCnt;
}

}
