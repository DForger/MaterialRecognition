#include "material_classifier.h"
#include <dirent.h>

MaterialClassifier::MaterialClassifier()
    : m_nFilterKernelWidth(7),
      m_nFilterKernelSetSize(64)
{

}

MaterialClassifier::~MaterialClassifier()
{

}

void MaterialClassifier::train(string dataAddress, MaterialParam &param)
{
    //if need build dictioary;
    if(param.useChroma){
        if(param.buildChromaDictionary){
            //build dictionary;
        }else{
            //load dictionary
        }
    }

    if(param.useSIFT){
        if(param.buildSIFTDictionary){
            //build sift dictionary
            buildSIFTDictionary(dataAddress);
        }else{
            //load dictioanry
            cv::Mat siftDictionary;
            Miscellaneous::IO::readData_<float>(siftDictionary, dataAddress+"/sift_dictionary");
            m_siftDictionary = siftDictionary;
        }
    }

    if(param.useTexton){
        if(param.buildFilterBank){
            buildFilterKernelSet(dataAddress);
        }else{
            cv::Mat filterBank;
            Miscellaneous::IO::readData_<float>(filterBank, dataAddress+"/filter_kernel_set");
            m_nFilterKernelSetSize = filterBank.rows;
            m_nFilterKernelWidth = std::sqrt(filterBank.cols);
            m_vecFilterKernelSet.clear();

            for(int i = 0 ; i < m_nFilterKernelSetSize; ++i){
                cv::Mat filter = cv::Mat_<float>(m_nFilterKernelWidth, m_nFilterKernelWidth);
                for(int j = 0; j < m_nFilterKernelWidth; ++j){
                    for(int k = 0; k < m_nFilterKernelWidth; ++k){
                        filter.at<float>(j, k) = filterBank.at<float>(i, j*m_nFilterKernelWidth+k);
                    }
                }
                m_vecFilterKernelSet.push_back(filter);
            }
        }

        if(param.buildTextonDictionary){
            buildGlobalTextonDictionary(dataAddress);
        }else{
            //load
            cv::Mat textonDictionary;
            Miscellaneous::IO::readData_<float>(textonDictionary, dataAddress+"/texton_dictionary");
            m_globalTextonDictionary = textonDictionary;
        }
    }


    //start extracting feature
    int nMaxSingleClassSize = 100;
    int nMaxDataSize = 800;
    std::vector<cv::Mat> vecSiftBoWSet;
    std::vector<cv::Mat> vecTextonBoWSet;
    std::vector<cv::Mat> vecChromaBoWSet;
    std::vector<cv::Mat> vecLabelSet;

    cv::Mat siftBowTrainDataSet = cv::Mat_<float>(nMaxDataSize, ms_nSiftDictionarySize);
    cv::Mat textonBowTrainDataSet = cv::Mat_<float>(nMaxDataSize, ms_nGlobalTextonDictionarySize);
    cv::Mat chromaBowTrainDataSet = cv::Mat_<float>(nMaxDataSize, ms_nChromaDictionarySize);
    cv::Mat labelTrainDataSet = cv::Mat_<float>(nMaxDataSize, 1);

    cv::Mat siftBowTestDataSet = cv::Mat_<float>(nMaxDataSize, ms_nSiftDictionarySize);
    cv::Mat textonBowTestDataSet = cv::Mat_<float>(nMaxDataSize, ms_nGlobalTextonDictionarySize);
    cv::Mat chromaBowTestDataSet = cv::Mat_<float>(nMaxDataSize, ms_nChromaDictionarySize);
    cv::Mat labelTestDataSet = cv::Mat_<float>(nMaxDataSize, 1);


    if(!param.useComputeFeatureSet){
        std::map<int, std::string> mapIndex2FileDirectory;
        mapIndex2FileDirectory.insert(std::pair<int, std::string>(0,"/fabric/"));
        mapIndex2FileDirectory.insert(std::pair<int, std::string>(1,"/foliage/"));
        mapIndex2FileDirectory.insert(std::pair<int, std::string>(2,"/glass/"));
        mapIndex2FileDirectory.insert(std::pair<int, std::string>(3,"/leather/"));
        mapIndex2FileDirectory.insert(std::pair<int, std::string>(4,"/metal/"));
        mapIndex2FileDirectory.insert(std::pair<int, std::string>(5,"/paper/"));
        mapIndex2FileDirectory.insert(std::pair<int, std::string>(6,"/plastic/"));
        mapIndex2FileDirectory.insert(std::pair<int, std::string>(7,"/stone/"));
        mapIndex2FileDirectory.insert(std::pair<int, std::string>(8,"/water/"));
        mapIndex2FileDirectory.insert(std::pair<int, std::string>(9,"/wood/"));


        int nTestDataCnt = 0;
        int nTrainDataCnt = 0;
        int nTrainDataSize = 50;
        for(int i = 0; i  < mapIndex2FileDirectory.size(); ++i){
            std::string strImgDirAddress = dataAddress+"/image"+mapIndex2FileDirectory[i];
            std::string strMaskDirAddress = dataAddress+"/mask"+mapIndex2FileDirectory[i];
            DIR *pDir = opendir(strImgDirAddress.c_str());
            if(pDir == NULL){
                continue;
            }

            cv::Mat labelSet = cv::Mat_<float>(nMaxSingleClassSize, 1, float(i));
            cv::Mat siftBoWSet = cv::Mat_<float>(nMaxSingleClassSize, ms_nSiftDictionarySize);
            cv::Mat textonBoWSet = cv::Mat_<float>(nMaxSingleClassSize, ms_nGlobalTextonDictionarySize);
            cv::Mat chromaBoWSet = cv::Mat_<float>(nMaxSingleClassSize, ms_nChromaDictionarySize);

            struct dirent *dirIter = readdir(pDir);
            int nImgCnt = 0;
            while(dirIter != NULL){
                std::string fileName(dirIter->d_name);
                dirIter = readdir(pDir);

                std::string imgAddress = strImgDirAddress +fileName;
                std::string maskAddress = strMaskDirAddress + fileName;

                std::string imgSiftSaveAddress = imgAddress;
                std::string imgTextonSaveAddress = imgAddress;
                int nDotIndex = imgSiftSaveAddress.find_last_of('.');
                int nLength = imgSiftSaveAddress.length();
                imgSiftSaveAddress.erase(imgSiftSaveAddress.begin()+nDotIndex, imgSiftSaveAddress.begin()+nLength-1);
                imgSiftSaveAddress.append("_sift_bow");

                imgTextonSaveAddress.erase(imgTextonSaveAddress.begin()+ nDotIndex, imgTextonSaveAddress.begin()+nLength-1);
                imgTextonSaveAddress.append("_texton_bow");

                cv::Mat img = cv::imread(imgAddress);
                cv::Mat mask = cv::imread(maskAddress, IMREAD_GRAYSCALE);
                if(img.empty()){
                    continue;
                }

                std::cout<<"computing "<<imgAddress<<std::endl;


                if(param.useSIFT){
                    cv::Mat siftDist;
                    extractSIFTDist(img, mask, siftDist);
                    Miscellaneous::IO::data2Text_<float>(siftDist, imgSiftSaveAddress);

                    float *srcPtr = siftDist.ptr<float>(0);
                    float *dstPtr = siftBoWSet.ptr<float>(nImgCnt);
                    for(int k = 0; k < ms_nSiftDictionarySize; ++k){
                        *dstPtr = *srcPtr;
                        ++dstPtr;
                        ++srcPtr;
                    }

                    //80 for training and 20 for testing
                    if(nImgCnt >= nTrainDataSize){
                        srcPtr = siftDist.ptr<float>(0);
                        dstPtr = siftBowTestDataSet.ptr<float>(nTestDataCnt);
                        for(int k = 0; k < ms_nSiftDictionarySize; ++k){
                            *dstPtr = *srcPtr;
                            ++dstPtr;
                            ++srcPtr;
                        }
                    }else{
                        srcPtr = siftDist.ptr<float>(0);
                        dstPtr = siftBowTrainDataSet.ptr<float>(nTrainDataCnt);
                        for(int k = 0; k < ms_nSiftDictionarySize; ++k){
                            *dstPtr = *srcPtr;
                            ++dstPtr;
                            ++srcPtr;
                        }
                    }
                }


                if(param.useTexton){
                    cv::Mat textonDist;
                    extractTextonDist(img, mask, textonDist);
                    Miscellaneous::IO::data2Text_<float>(textonDist, imgTextonSaveAddress);
                    float *srcPtr = textonDist.ptr<float>(0);
                    float *dstPtr = textonBoWSet.ptr<float>(nImgCnt);
                    for(int k = 0; k < ms_nGlobalTextonDictionarySize; ++k){
                        *dstPtr = *srcPtr;
                        ++dstPtr;
                        ++srcPtr;
                    }

                    if(nImgCnt >= nTrainDataSize){
                        srcPtr = textonDist.ptr<float>(0);
                        dstPtr = textonBowTestDataSet.ptr<float>(nTestDataCnt);
                        for(int k = 0; k < ms_nGlobalTextonDictionarySize; ++k){
                            *dstPtr = *srcPtr;
                            ++dstPtr;
                            ++srcPtr;
                        }
                    }else{
                        srcPtr = textonDist.ptr<float>(0);
                        dstPtr = textonBowTrainDataSet.ptr<float>(nTrainDataCnt);
                        for(int k = 0; k < ms_nGlobalTextonDictionarySize; ++k){
                            *dstPtr = *srcPtr;
                            ++dstPtr;
                            ++srcPtr;
                        }
                    }

                }

                if(nImgCnt >= nTrainDataSize){
                    labelTestDataSet.at<float>(nTestDataCnt,0) = i;
                    ++nTestDataCnt;
                }else{
                    labelTrainDataSet.at<float>(nTrainDataCnt, 0) = i;
                    ++nTrainDataCnt;
                }
                ++nImgCnt;
            }

            if(param.useSIFT){
                siftBoWSet.adjustROI(0, nMaxSingleClassSize-nImgCnt, 0, 0);
                vecSiftBoWSet.push_back(siftBoWSet);
            }

            if(param.useTexton){
                textonBoWSet.adjustROI(0, nMaxSingleClassSize-nImgCnt, 0, 0);
                vecTextonBoWSet.push_back(textonBoWSet);
            }

            if(param.useChroma){
                chromaBoWSet.adjustROI(0, nMaxSingleClassSize-nImgCnt, 0, 0);
                vecChromaBoWSet.push_back(chromaBoWSet);
            }

            labelSet.adjustROI(0, nMaxSingleClassSize-nImgCnt, 0, 0);
            vecLabelSet.push_back(labelSet);

            closedir(pDir);
        }
        labelTrainDataSet.adjustROI(0,  nTrainDataCnt-nMaxDataSize, 0, 0);
        siftBowTrainDataSet.adjustROI(0, nTrainDataCnt-nMaxDataSize, 0, 0);
        textonBowTrainDataSet.adjustROI(0, nTrainDataCnt-nMaxDataSize, 0, 0);
        chromaBowTrainDataSet.adjustROI(0, nTrainDataCnt-nMaxDataSize, 0, 0);

        labelTestDataSet.adjustROI(0, nTestDataCnt-nMaxDataSize, 0, 0);
        siftBowTestDataSet.adjustROI(0, nTestDataCnt-nMaxDataSize, 0, 0);
        textonBowTestDataSet.adjustROI(0, nTestDataCnt-nMaxDataSize, 0, 0);
        chromaBowTestDataSet.adjustROI(0, nTestDataCnt-nMaxDataSize, 0, 0);

        Miscellaneous::IO::data2Text_<float>(labelTrainDataSet, dataAddress+"/train_label_set");
        Miscellaneous::IO::data2Text_<float>(labelTestDataSet, dataAddress+"/test_label_set");

        if(param.useSIFT){
            Miscellaneous::IO::data2Text_<float>(siftBowTrainDataSet, dataAddress+"/org_sift_bow_train_data");
            Miscellaneous::IO::data2Text_<float>(siftBowTestDataSet, dataAddress+"/org_sift_bow_test_data");
        }

        if(param.useTexton){
            Miscellaneous::IO::data2Text_<float>(textonBowTrainDataSet, dataAddress+"/org_texton_bow_train_data");
            Miscellaneous::IO::data2Text_<float>(textonBowTestDataSet, dataAddress+"/org_texton_bow_test_data");
        }

    }else{
        Miscellaneous::IO::readData_<float>(labelTrainDataSet, dataAddress+"/train_label_set");
        Miscellaneous::IO::readData_<float>(labelTestDataSet, dataAddress+"/test_label_set");

        if(param.useSIFT){
            Miscellaneous::IO::readData_<float>(siftBowTrainDataSet, dataAddress+"/org_sift_bow_train_data");
            Miscellaneous::IO::readData_<float>(siftBowTestDataSet, dataAddress+"/org_sift_bow_test_data");
        }

        if(param.useTexton){
            Miscellaneous::IO::readData_<float>(textonBowTrainDataSet, dataAddress+"/org_texton_bow_train_data");
            Miscellaneous::IO::readData_<float>(textonBowTestDataSet, dataAddress+"/org_texton_bow_test_data");
        }
    }


    //combine features to do lda
    cv::LDA lda;
    cv::PCA pca;
    cv::Mat projectedsiftBowTrainDataSet;
    cv::Mat projectedtextonBowTrainDataSet;
    cv::Mat projectedchromaBowTrainDataSet;

    cv::Mat label;
    labelTrainDataSet.convertTo(label, CV_32SC1);

    if(param.useSIFT){
        lda.compute(siftBowTrainDataSet, labelTrainDataSet);
        cv::Mat siftEigenVector = lda.eigenvectors();
        siftEigenVector.convertTo(siftEigenVector, CV_32F);
        Miscellaneous::IO::data2Text_<float>(siftEigenVector,dataAddress+"/sift_eigen");
        projectedsiftBowTrainDataSet = siftBowTrainDataSet*siftEigenVector;

    }

    if(param.useTexton){
        lda.compute(textonBowTrainDataSet, labelTrainDataSet);
        cv::Mat textonEigenVector = lda.eigenvectors();
        textonEigenVector.convertTo(textonEigenVector, CV_32F);
        Miscellaneous::IO::data2Text_<float>(textonEigenVector, dataAddress+"/texton_eigen");
        projectedtextonBowTrainDataSet = textonBowTrainDataSet*textonEigenVector;
    }



    //    cv::Mat combined = cv::Mat_<float>(projectedtextonBowTrainDataSet.rows, projectedtextonBowTrainDataSet.cols*2);
    //    combined.adjustROI(0, 0, 0, -9);
    //    projectedtextonBowTrainDataSet.copyTo(combined);
    //    combined.adjustROI(0, 0, -9, 9);
    //    projectedsiftBowTrainDataSet.copyTo(combined);
    //    combined.adjustROI(0, 0 , 9, 0);


    cv::ml::KNearest::Params knnParam;
    knnParam.isclassifier = true;
    knnParam.defaultK = 30;
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::StatModel::train<cv::ml::KNearest>(siftBowTrainDataSet, cv::ml::ROW_SAMPLE, label, knnParam);

    cv::ml::SVM::Params svmParams;

    double epsilon = 1e-7;
    svmParams.svmType = cv::ml::SVM::C_SVC;
    svmParams.kernelType = cv::ml::SVM::RBF;
    svmParams.degree = 3;
    svmParams.gamma = 0.01;
    svmParams.C = 0.1;
    svmParams.termCrit = cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 100000, epsilon);

    cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::train<cv::ml::SVM>(siftBowTrainDataSet, cv::ml::ROW_SAMPLE, label, svmParams);
    cv::Mat sv = svm->getSupportVectors();
    std::cout<<"support vector "<<sv.rows<<std::endl;
    std::cout.flush();
    svm->save("sift.xml");

    //compute E_in
    cv::Mat predictedTrainLabel;
    svm->predict(siftBowTrainDataSet, predictedTrainLabel);
//    knn->findNearest(siftBowTrainDataSet, 30, predictedTrainLabel);


    double Ein = 0;
    for(int i = 0; i < predictedTrainLabel.rows; ++i){
        if(std::fabs(predictedTrainLabel.at<float>(i,0) - labelTrainDataSet.at<float>(i,0)) > 0.5){
            Ein += 1;
        }
    }
    Ein = Ein/predictedTrainLabel.rows;

    std::cout<<"Trained Ein is "<<Ein<<std::endl;
    std::cout.flush();

    //compute E_out
    cv::Mat predictedTestLabel;
    svm->predict(siftBowTestDataSet, predictedTestLabel);

//    knn->findNearest(siftBowTestDataSet, 30, predictedTestLabel);

    double Eout = 0;
    for(int i = 0; i < predictedTestLabel.rows; ++i){
        if(std::fabs(predictedTestLabel.at<float>(i,0) - labelTestDataSet.at<float>(i,0)) > 0.5){
            Eout += 1;
        }
    }
    Eout = Eout/predictedTestLabel.rows;

    std::cout<<"Trained Eout is "<<Eout<<std::endl;
    std::cout.flush();

}

void MaterialClassifier::load(string paraAddress)
{

}

void MaterialClassifier::test(Mat img)
{

}

void MaterialClassifier::extractClusterFeature(Mat img, SLICSuperpixel &slic, std::vector<MaterialFeatureSet> &clusterFeatureSet)
{

}

void MaterialClassifier::extractTextonDist(cv::Mat img, cv::Mat mask,  cv::Mat &textonDist)
{
    cv::Mat f_grayImg, grayImg;
    if(img.channels() == 1){
        grayImg = img.clone();
        grayImg.convertTo(f_grayImg, CV_32F);
    }else{
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
        grayImg.convertTo(f_grayImg, CV_32F);
    }

    int nRow = img.rows;
    int nCol = img.cols;


    std::vector<cv::Mat> responseMap(m_nFilterKernelSetSize);
    cv::Mat totalResponseMat;
    for(int j = 0; j < m_nFilterKernelSetSize; ++j){
        cv::filter2D(f_grayImg, responseMap[j], CV_32F, m_vecFilterKernelSet[j]);
    }
    cv::merge(responseMap, totalResponseMat);
    /*
        visualize response map
                for(int i = 0; i < responseMap.size(); ++i){
                    QString fileName = QString::number(i)+"_response.tiff";
                    cv::Mat visualImg = responseMap[i].clone();
                    visualImg = cv::abs(visualImg);
                    double min,max;
                    cv::minMaxLoc(visualImg,&min, &max);
                    cv::Mat saveImg;
                    visualImg = 255*(visualImg - min)/(max-min);
                    visualImg.convertTo(saveImg,CV_8U);
                    cv::imwrite(fileName.toLocal8Bit().data(), saveImg);
                }
                responseVec = cv::Mat_<float>(totalResponseMat.rows*totalResponseMat.cols, totalResponseMat.channels());
                for(int j = 0; j < totalResponseMat.rows; ++j){
                    float *srcPtr = totalResponseMat.ptr<float>(j);
                    for(int k = 0; k < totalResponseMat.cols; ++k){
                        float *dstPtr = responseVec.ptr<float>(j*n_col+k);
                        for(int l = 0; l < totalResponseMat.channels(); ++l){
                            *dstPtr++ = *srcPtr++;
                        }
                    }
                }
        */

    float voteBins[ms_nGlobalTextonDictionarySize];
    double voteSum = 0.01;
    std::fill(voteBins, voteBins+ms_nGlobalTextonDictionarySize, float(0));

    uchar *maskPtr;
    for(int i = 0; i < nRow;i+= 5 ){
        maskPtr = mask.ptr<uchar>(i);
        for(int k = 0; k < nCol; k+=5){
            if((*maskPtr)< 10){
                ++maskPtr;
                continue;
            }else{
                ++maskPtr;
            }
            float *basePtr = totalResponseMat.ptr<float>(i)+k*m_nFilterKernelSetSize;
            float *responsePtr = basePtr;

            //normalize with l2 norm
            float l2Norm = 0.01, scale;
            for(int l = 0; l < m_nFilterKernelSetSize; ++l){
                l2Norm = l2Norm + (*responsePtr)*(*responsePtr);
                ++responsePtr;
            }
            scale = std::log((1+l2Norm)/0.03)/l2Norm;

            responsePtr = basePtr;
            for(int l = 0; l < m_nFilterKernelSetSize; ++l){
                *responsePtr = (*responsePtr)*scale;
                ++responsePtr;
            }

            // compare and soft vote
            responsePtr = basePtr;
            cv::Mat testVector = cv::Mat_<float>(1, m_nFilterKernelSetSize, responsePtr);

            for(int j = 0; j < ms_nGlobalTextonDictionarySize; ++j){
                double correl = cv::compareHist(testVector, m_globalTextonDictionary.row(j), cv::HISTCMP_CORREL);
                voteSum = voteSum + correl;
                voteBins[j] = voteBins[j] + correl;
            }
        }
    }

    cv::Mat textonHistogram = cv::Mat_<float>(1, ms_nGlobalTextonDictionarySize);
    float *textonHistPtr = textonHistogram.ptr<float>(0);
    float scale = 1.0/voteSum;      //l1 normalize factor
    for(int i = 0; i < ms_nGlobalTextonDictionarySize; ++i){
        *textonHistPtr = voteBins[i]*scale;
        ++textonHistPtr;
    }

    textonDist = textonHistogram;
}

void MaterialClassifier::extractSIFTDist(Mat img, Mat mask, Mat &siftDist)
{
    int nRow = img.rows;
    int nCol = img.cols;
    float imgArray[nRow*nCol];
    //convert to vlfeat's image format
    for(int k = 0; k < nRow; ++k){
        uchar *srcPtr = img.ptr<uchar>(k);
        float *dstPtr = imgArray + k*nCol;
        for(int l = 0; l < nCol; ++l){
            *dstPtr = (float)(*srcPtr);
            ++dstPtr;
            ++srcPtr;
        }
    }

    //detect dense sift
    VlDsiftFilter *dsift = vl_dsift_new_basic(nCol, nRow, 5, 16);
    vl_dsift_process(dsift, imgArray);
    const float *fpDescriptorPtr = vl_dsift_get_descriptors(dsift);
    int nDescriptorSize = vl_dsift_get_descriptor_size(dsift);
    int nKeyPointNum = vl_dsift_get_keypoint_num(dsift);
    const VlDsiftKeypoint *keyPoints =  vl_dsift_get_keypoints(dsift);

    cv::Mat siftDescriptor = cv::Mat_<float>(1, nDescriptorSize);

    cv::Mat siftHist = cv::Mat_<float>(1, ms_nSiftDictionarySize);
    float votedBin[ms_nSiftDictionarySize];
    std::fill(votedBin, votedBin+ms_nSiftDictionarySize, float(0));
    double voteSum = 0;
    for(int i = 0; i < nKeyPointNum; ++i){
        int x = keyPoints[i].x;
        int y = keyPoints[i].y;
        if(mask.at<uchar>(y,x) < 100){
            continue;
        }

        float *dstPtr = siftDescriptor.ptr<float>(0);
        for(int j = 0; j < nDescriptorSize; ++j){
            *dstPtr = fpDescriptorPtr[i*nDescriptorSize+j];
            ++dstPtr;
        }
        for(int j = 0; j < ms_nSiftDictionarySize; ++j){
            double correl = cv::compareHist(siftDescriptor, m_siftDictionary.row(j), cv::HISTCMP_CORREL);
            votedBin[j] += correl;
            voteSum = voteSum + correl;
        }
    }

    //normalize
    float *siftHistPtr = siftHist.ptr<float>(0);
    for(int i = 0; i < ms_nSiftDictionarySize; ++i){
        *siftHistPtr = votedBin[i]/voteSum;
        ++siftHistPtr;
    }

    siftDist = siftHist;

    vl_dsift_delete(dsift);
}

void MaterialClassifier::buildFilterKernelSet(string dataAddress)
{
    std::map<int, std::string> mapIndex2FileDirectory;
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(0,"/fabric/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(1,"/foliage/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(2,"/glass/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(3,"/leather/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(4,"/metal/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(5,"/paper/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(6,"/plastic/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(7,"/stone/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(8,"/water/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(9,"/wood/"));


    int nKernelLength = m_nFilterKernelWidth * m_nFilterKernelWidth;
    int nHalfKernelSize = m_nFilterKernelWidth/2;

    int shiftX[nKernelLength];
    int shiftY[nKernelLength];

    for(int i = 0; i < m_nFilterKernelWidth; ++i){
        for(int j = 0; j < m_nFilterKernelWidth; ++j){
            shiftX[i*m_nFilterKernelWidth+j] = i- nHalfKernelSize;
            shiftY[i*m_nFilterKernelWidth+j] = j - nHalfKernelSize;
        }
    }

    int nMaxDataSize = 500000;
    cv::Mat patchSet;
    patchSet = cv::Mat_<float>(nMaxDataSize, nKernelLength);
    int nDataCnt = 0;
    for(int i = 0; i  < mapIndex2FileDirectory.size(); ++i){
        std::string strImgDirAddress = dataAddress+"/image"+mapIndex2FileDirectory[i];
        std::string strMaskDirAddress = dataAddress+"/mask"+mapIndex2FileDirectory[i];
        DIR *pDir = opendir(strImgDirAddress.c_str());
        if(pDir == NULL){
            continue;
        }
        struct dirent *dirIter = readdir(pDir);
        int nImgCnt = 0;
        while(dirIter != NULL){
            std::string fileName(dirIter->d_name);
            dirIter = readdir(pDir);

            std::string imgAddress = strImgDirAddress +fileName;
            std::string maskAddress = strMaskDirAddress + fileName;

            cv::Mat img = cv::imread(imgAddress);
            cv::Mat mask = cv::imread(maskAddress, IMREAD_GRAYSCALE);
            if(img.empty()){
                continue;
            }
            if(nImgCnt > 80){
                continue;
            }

            ++nImgCnt;

            cv::Mat grayImg;
            cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

            int nCol = img.cols;
            int nRow = img.rows;

            int nMaxX = nCol - nHalfKernelSize;
            int nMinX = nHalfKernelSize;
            int nMaxY = nRow - nHalfKernelSize;
            int nMinY = nHalfKernelSize;
            for(int y = nMinY; y < nMaxY; y+=5){
                for(int x = nMinX; x < nMaxX; x+=5){
                    uchar maskValue = mask.at<uchar>(y, x);
                    if(maskValue <= 1){
                        continue;
                    }
                    float lucky = std::rand()*1.0/RAND_MAX;
                    if(lucky > 0.1){
                        continue;
                    }

                    if(nDataCnt >= nMaxDataSize){
                        continue;
                    }

                    float *dstPtr = patchSet.ptr<float>(nDataCnt);
                    for(int k = 0; k < nKernelLength; ++k){
                        *dstPtr = grayImg.at<uchar>(y+shiftY[k], x+shiftX[k]);
                        ++dstPtr;
                    }

                    ++nDataCnt;
                }
            }
        }

        closedir(pDir);
    }

    patchSet.adjustROI(0,nDataCnt-nMaxDataSize,0,0);
    int K = m_nFilterKernelSetSize;
    cv::TermCriteria critera;
    critera.type = cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS;
    critera.maxCount = 500;
    critera.epsilon = 0.1;
    cv::Mat bestLabels;
    cv::Mat centers;
    cv::kmeans(patchSet, K, bestLabels, critera, 20, cv::KMEANS_PP_CENTERS, centers);

    //visualize
    for(int i = 0; i < centers.rows; ++i){
        cv::Mat kernelImg = cv::Mat_<uchar>(m_nFilterKernelWidth,m_nFilterKernelWidth);
        double min, max;
        cv::minMaxLoc(centers.row(i), &min, &max);
        float scale = 254/(max-min);
        for(int j = 0; j < m_nFilterKernelWidth; ++j){
            for(int k = 0; k < m_nFilterKernelWidth; ++k){
                kernelImg.at<uchar>(j,k) = static_cast<uchar>(scale*centers.at<float>(i, j*m_nFilterKernelWidth+k));
            }
        }
        QString name = QString::number(i);
        cv::imshow(name.toLocal8Bit().data(), kernelImg);
    }

    //zero mean and scale
    for(int i = 0; i < centers.rows; ++i){
        cv::Mat rowVec = centers.row(i);
        double l1Norm, shift;

        cv::Scalar meanValue = cv::mean(rowVec);
        shift = meanValue[0];

        rowVec  = rowVec-shift;
        l1Norm = 0;
        for(int j = 0; j < centers.cols; ++j){
            l1Norm += std::fabs(centers.at<float>(i,j));
        }

        rowVec = rowVec/l1Norm;
    }

    if(m_vecFilterKernelSet.size() != 0){
        m_vecFilterKernelSet.clear();
    }
    m_vecFilterKernelSet.resize(m_nFilterKernelSetSize);

    for(size_t i = 0; i < centers.rows; ++i){
        m_vecFilterKernelSet[i] = cv::Mat_<float>(m_nFilterKernelWidth, m_nFilterKernelWidth);
        for(int j = 0; j < m_nFilterKernelWidth; ++j){
            for(int k = 0; k < m_nFilterKernelWidth; ++k){
                (m_vecFilterKernelSet[i]).at<float>(j,k) = centers.at<float>(i, j*m_nFilterKernelWidth+k);
            }
        }
    }
    Miscellaneous::IO::data2Text_<float>(centers, dataAddress+"/filter_kernel_set");

    cv::waitKey(0);
}

void MaterialClassifier::buildGlobalTextonDictionary(string dataAddress)
{
    std::map<int, std::string> mapIndex2FileDirectory;
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(0,"/fabric/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(1,"/foliage/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(2,"/glass/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(3,"/leather/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(4,"/metal/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(5,"/paper/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(6,"/plastic/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(7,"/stone/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(8,"/water/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(9,"/wood/"));

    int nHalfKernelSize = m_nFilterKernelWidth/2;

    int nMaxDataSize = 500000;
    cv::Mat filterResponseSet;
    filterResponseSet = cv::Mat_<float>(nMaxDataSize, m_nFilterKernelSetSize);
    int nDataCnt = 0;
    for(int i = 0; i  < mapIndex2FileDirectory.size(); ++i){
        std::string strImgDirAddress = dataAddress+"/image"+mapIndex2FileDirectory[i];
        std::string strMaskDirAddress = dataAddress+"/mask"+mapIndex2FileDirectory[i];
        DIR *pDir = opendir(strImgDirAddress.c_str());
        if(pDir == NULL){
            continue;
        }
        struct dirent *dirIter = readdir(pDir);
        int nImgCnt = 0;
        while(dirIter != NULL){
            std::string fileName(dirIter->d_name);
            dirIter = readdir(pDir);

            std::string imgAddress = strImgDirAddress +fileName;
            std::string maskAddress = strMaskDirAddress + fileName;

            cv::Mat img = cv::imread(imgAddress);
            cv::Mat mask = cv::imread(maskAddress, IMREAD_GRAYSCALE);
            if(img.empty()){
                continue;
            }
            //80 for training and 20 for testing
            if(nImgCnt > 80){
                continue;
            }

            ++nImgCnt;

            cv::Mat totalResponse;
            //compute filtering response
            {
                cv::Mat grayImg, f_grayImg;
                cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
                grayImg.convertTo(f_grayImg, CV_32F);

                std::vector<cv::Mat> responseMap(m_nFilterKernelSetSize);
                for(int j = 0; j < m_nFilterKernelSetSize; ++j){
                    cv::filter2D(f_grayImg, responseMap[j], CV_32F, m_vecFilterKernelSet[j]);
                }

                cv::merge(responseMap, totalResponse);
            }

            //sample response value
            int nCol = img.cols;
            int nRow = img.rows;
            int nMaxX = nCol - nHalfKernelSize;
            int nMinX = nHalfKernelSize;
            int nMaxY = nRow - nHalfKernelSize;
            int nMinY = nHalfKernelSize;
            for(int y = nMinY; y < nMaxY; y+=5){
                for(int x = nMinX; x < nMaxX; x+=5){
                    uchar maskValue = mask.at<uchar>(y, x);
                    if(maskValue <= 1){
                        continue;
                    }
                    float lucky = std::rand()*1.0/RAND_MAX;
                    if(lucky > 0.05){
                        continue;
                    }

                    if(nDataCnt >= nMaxDataSize){
                        break;
                    }
                    //normalize
                    float l2Norm = 0.01, scale;
                    float *srcPtr, *dstPtr;
                    srcPtr= totalResponse.ptr<float>(y) + x*m_nFilterKernelSetSize;
                    for(int k = 0; k < m_nFilterKernelSetSize; ++k){
                        l2Norm = l2Norm + (*srcPtr)*(*srcPtr);
                        ++srcPtr;
                    }
                    scale = std::log((1+l2Norm)/0.03)/l2Norm;

                    srcPtr = totalResponse.ptr<float>(y) + x*m_nFilterKernelSetSize;
                    dstPtr = filterResponseSet.ptr<float>(nDataCnt);
                    for(int k = 0; k < m_nFilterKernelSetSize; ++k){
                        *dstPtr = (*srcPtr)*scale;
                        ++dstPtr;
                        ++srcPtr;
                    }

                    ++nDataCnt;
                }
            }
        }
        closedir(pDir);
    }

    //k-mean cluster to build texton

    filterResponseSet.adjustROI(0,nDataCnt-nMaxDataSize,0,0);
    int K = ms_nGlobalTextonDictionarySize;
    cv::TermCriteria critera;
    critera.type = cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS;
    critera.maxCount = 500;
    critera.epsilon = 0.1;
    cv::Mat bestLabels;
    cv::Mat centers;
    cv::kmeans(filterResponseSet, K, bestLabels, critera, 20, cv::KMEANS_PP_CENTERS, centers);

    m_globalTextonDictionary = centers;

    Miscellaneous::IO::data2Text_<float>(centers, dataAddress+"/texton_dictionary");
}

void MaterialClassifier::buildSIFTDictionary(string dataAddress)
{
    std::map<int, std::string> mapIndex2FileDirectory;
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(0,"/fabric/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(1,"/foliage/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(2,"/glass/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(3,"/leather/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(4,"/metal/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(5,"/paper/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(6,"/plastic/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(7,"/stone/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(8,"/water/"));
    mapIndex2FileDirectory.insert(std::pair<int, std::string>(9,"/wood/"));


    int nMaxDataSize = 3000000;
    int nSiftDim = 128;
    cv::Mat siftDescriptorSet = cv::Mat_<float>(nMaxDataSize, nSiftDim);
    int nDataCnt = 0;
    for(int i = 0; i  < mapIndex2FileDirectory.size(); ++i){
        std::string strImgDirAddress = dataAddress+"/image"+mapIndex2FileDirectory[i];
        std::string strMaskDirAddress = dataAddress+"/mask"+mapIndex2FileDirectory[i];
        DIR *pDir = opendir(strImgDirAddress.c_str());
        if(pDir == NULL){
            continue;
        }
        struct dirent *dirIter = readdir(pDir);
        int nImgCnt = 0;
        while(dirIter != NULL){
            std::string fileName(dirIter->d_name);
            dirIter = readdir(pDir);

            std::string imgAddress = strImgDirAddress +fileName;
            std::string maskAddress = strMaskDirAddress + fileName;

            std::string imgSiftSaveAddress = imgAddress;
            int nDotIndex = imgSiftSaveAddress.find_last_not_of('.');
            imgSiftSaveAddress.erase(imgSiftSaveAddress.begin()+nDotIndex, imgSiftSaveAddress.begin()+imgSiftSaveAddress.length()-1);
            imgSiftSaveAddress.append("_sift_descriptor");

            cv::Mat img = cv::imread(imgAddress, IMREAD_GRAYSCALE);
            cv::Mat mask = cv::imread(maskAddress, IMREAD_GRAYSCALE);
            if(img.empty()){
                continue;
            }
            //50 for training and 50 for testing
            if(nImgCnt > 50){
                continue;
            }
            ++nImgCnt;

            int nRow = img.rows;
            int nCol = img.cols;
            float imgArray[nRow*nCol];
            //convert to vlfeat's image format
            for(int k = 0; k < nRow; ++k){
                uchar *srcPtr = img.ptr<uchar>(k);
                float *dstPtr = imgArray + k*nCol;
                for(int l = 0; l < nCol; ++l){
                    *dstPtr = (float)(*srcPtr);
                    ++dstPtr;
                    ++srcPtr;
                }
            }

            //detect dense sift
            VlDsiftFilter *dsift = vl_dsift_new_basic(nCol, nRow, 5, 16);

            vl_dsift_process(dsift, imgArray);
            const float *fpDescriptorPtr = vl_dsift_get_descriptors(dsift);
            int nDescriptorSize = vl_dsift_get_descriptor_size(dsift);
            int nKeyPointNum = vl_dsift_get_keypoint_num(dsift);
            const VlDsiftKeypoint *keyPoints =  vl_dsift_get_keypoints(dsift);

            for(int k = 0; k < nKeyPointNum; ++k){
                int x = keyPoints[k].x;
                int y = keyPoints[k].y;
                if(mask.at<uchar>(y,x) < 100){
                    continue;
                }
                //add to data set
                float *dstPtr = siftDescriptorSet.ptr<float>(nDataCnt);
                for(int l = 0; l < nDescriptorSize; ++l){
                    *dstPtr = fpDescriptorPtr[k*nDescriptorSize+l];
                    ++dstPtr;
                }
                ++nDataCnt;
            }
            vl_dsift_delete(dsift);
            //            Miscellaneous::IO::data2Text_<float>(siftDescriptor, imgSiftSaveAddress);
        }
        closedir(pDir);
    }

    //k-mean cluster to build texton
    siftDescriptorSet.adjustROI(0,nDataCnt-nMaxDataSize,0,0);
    int K = ms_nSiftDictionarySize;
    cv::TermCriteria critera;
    critera.type = cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS;
    critera.maxCount = 500;
    critera.epsilon = 0.1;
    cv::Mat bestLabels;
    cv::Mat centers;
    cv::kmeans(siftDescriptorSet, K, bestLabels, critera, 20, cv::KMEANS_PP_CENTERS, centers);

    m_siftDictionary = centers;

    Miscellaneous::IO::data2Text_<float>(centers, dataAddress+"/sift_dictionary");
}

void MaterialClassifier::buildChromaDictionary(string dataAddress)
{

}

void MaterialClassifier::buildColorModelSet(string dataAddress)
{

}

void MaterialClassifier::buildModelSet(string dataAddress)
{

}

size_t MaterialClassifier::ms_nGlobalTextonDictionarySize = 48;
size_t MaterialClassifier::ms_nChromaDictionarySize = 64;
size_t MaterialClassifier::ms_nSiftDictionarySize = 128;
