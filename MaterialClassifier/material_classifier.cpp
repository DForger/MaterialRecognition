#include "material_classifier.h"
#include <dirent.h>

MaterialClassifier::MaterialClassifier()
    : m_nFilterKernelWidth(7),
      m_nFilterKernelSetSize(96),
      m_nSiftBinNum(8)
{
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

    //    mapIndex2FileDirectory.insert(std::pair<int, std::string>(0,"/aluminium_foil/"));
    //    mapIndex2FileDirectory.insert(std::pair<int, std::string>(1,"/brown_bread/"));
    //    mapIndex2FileDirectory.insert(std::pair<int, std::string>(2,"/corduroy/"));
    //    mapIndex2FileDirectory.insert(std::pair<int, std::string>(3,"/cotton/"));
    //    mapIndex2FileDirectory.insert(std::pair<int, std::string>(4,"/cracker/"));
    //    mapIndex2FileDirectory.insert(std::pair<int, std::string>(5,"/linen/"));
    //    mapIndex2FileDirectory.insert(std::pair<int, std::string>(6,"/orange_peel/"));
    //    mapIndex2FileDirectory.insert(std::pair<int, std::string>(7,"/sandpaper/"));
    //    mapIndex2FileDirectory.insert(std::pair<int, std::string>(8,"/sponge/"));
    //    mapIndex2FileDirectory.insert(std::pair<int, std::string>(9,"/styrofoam/"));

    m_siftScale[0] = 1;
    m_siftScale[1] = 0.5;
    m_siftScale[2] = 0.25;
}

MaterialClassifier::~MaterialClassifier()
{

}

void MaterialClassifier::train(string dataAddress, MaterialParam &param)
{
    //if need build dictioary;
    if(param.useColorIFV){
        if(param.buildColorGmmDist){
            //build dictionary;
            buildColorGMMDist(dataAddress);
        }else{
            //load dictionary
            int patchDim = 27;
            m_colorGMMDist.load(std::string(dataAddress+"/color_gmm_info"), patchDim);
            ms_nColorIFVDimension = m_colorGMMDist.clusterNum*m_colorGMMDist.dimension*2;
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

    if(param.useSiftIFV){
        if(param.buildSIFTGmmDist){
            buildSiftIfvGMM(dataAddress);
            ms_nSiftIFVDimension = m_siftGMMDist.clusterNum*m_siftGMMDist.dimension*2;
        }else{
            int siftDim = 128;
            m_siftGMMDist.load(std::string(dataAddress+"/sift_gmm_info"), siftDim);
            ms_nSiftIFVDimension = m_siftGMMDist.clusterNum*m_siftGMMDist.dimension*2;
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

    if(param.useTextonIFV){
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

        if(param.buildTextonGmmDist){
            buildTextonGmm(dataAddress);
        }else{
            //load
            int filterSize = m_nFilterKernelSetSize;
            m_textonGMMDist.load(std::string(dataAddress+"/texton_gmm_info"), filterSize);
            ms_nTextonIFVDimension = m_textonGMMDist.clusterNum*m_textonGMMDist.dimension*2;
        }
    }


    //start extracting feature
    int nMaxSingleClassSize = 100;
    int nMaxDataSize = 800;
    std::vector<cv::Mat> vecSiftBoWSet;
    std::vector<cv::Mat> vecTextonBoWSet;
    std::vector<cv::Mat> vecColorBoWSet;
    std::vector<cv::Mat> vecLabelSet;

    cv::Mat siftBowTrainDataSet = cv::Mat_<float>(nMaxDataSize, ms_nSiftDictionarySize);
    cv::Mat textonBowTrainDataSet = cv::Mat_<float>(nMaxDataSize, ms_nGlobalTextonDictionarySize);
    cv::Mat colorIfvTrainDataSet = cv::Mat_<float>(nMaxDataSize, ms_nColorIFVDimension);
    cv::Mat siftIfvTrainDataSet = cv::Mat_<float>(nMaxDataSize, ms_nSiftIFVDimension);
    cv::Mat textonIfvTrainDataSet = cv::Mat_<float>(nMaxDataSize, ms_nTextonIFVDimension);
    cv::Mat labelTrainDataSet = cv::Mat_<float>(nMaxDataSize, 1);


    cv::Mat siftBowTestDataSet = cv::Mat_<float>(nMaxDataSize, ms_nSiftDictionarySize);
    cv::Mat textonBowTestDataSet = cv::Mat_<float>(nMaxDataSize, ms_nGlobalTextonDictionarySize);
    cv::Mat colorIfvTestDataSet = cv::Mat_<float>(nMaxDataSize, ms_nColorIFVDimension);
    cv::Mat siftIfvTestDataSet = cv::Mat_<float>(nMaxDataSize, ms_nSiftIFVDimension);
    cv::Mat textonIfvTestDataSet = cv::Mat_<float>(nMaxDataSize, ms_nTextonIFVDimension);
    cv::Mat labelTestDataSet = cv::Mat_<float>(nMaxDataSize, 1);


    if(!param.useComputeFeatureSet){

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
            cv::Mat colorBoWSet = cv::Mat_<float>(nMaxSingleClassSize, ms_nColorDictionarySize);


            struct dirent *dirIter = readdir(pDir);
            int nImgCnt = 0;
            while(dirIter != NULL){
                std::string fileName(dirIter->d_name);
                dirIter = readdir(pDir);

                std::string imgAddress = strImgDirAddress +fileName;
                std::string maskAddress = strMaskDirAddress + fileName;

                int nDotIndex = imgAddress.find_last_of('.');
                int nLength = imgAddress.length();

                cv::Mat img = cv::imread(imgAddress);
                cv::Mat mask = cv::imread(maskAddress, IMREAD_GRAYSCALE);
                if(img.empty()){
                    continue;
                }

                if(mask.empty()){
                    mask = cv::Mat_<uchar>(img.rows, img.cols, (uchar)255);
                }

                std::cout<<"computing "<<imgAddress<<std::endl;

                if(param.useColorIFV){
                    cv::Mat colorIfv;
                    extractColorIFV(img, mask, colorIfv, m_colorGMMDist);

                    std::string imgColorIfvSaveAddress = imgAddress;
                    imgColorIfvSaveAddress.erase(imgColorIfvSaveAddress.begin()+nDotIndex, imgColorIfvSaveAddress.begin()+nLength-1);
                    imgColorIfvSaveAddress.append("_color_ifv");
                    Miscellaneous::IO::data2Text_<float>(colorIfv, imgColorIfvSaveAddress);

                    float *srcPtr;
                    float *dstPtr ;
                    if(nImgCnt >= nTrainDataSize){
                        srcPtr = colorIfv.ptr<float>(0);
                        dstPtr = colorIfvTestDataSet.ptr<float>(nTestDataCnt);
                        for(int k = 0; k < ms_nColorIFVDimension; ++k){
                            *dstPtr = *srcPtr;
                            ++dstPtr;
                            ++srcPtr;
                        }
                    }else{
                        srcPtr = colorIfv.ptr<float>(0);
                        dstPtr = colorIfvTrainDataSet.ptr<float>(nTrainDataCnt);
                        for(int k = 0; k < ms_nColorIFVDimension; ++k){
                            *dstPtr = *srcPtr;
                            ++dstPtr;
                            ++srcPtr;
                        }
                    }
                }

                if(param.useSIFT){
                    cv::Mat siftDist;
                    extractSIFTDist(img, mask, siftDist);

                    std::string imgSiftSaveAddress = imgAddress;
                    imgSiftSaveAddress.erase(imgSiftSaveAddress.begin()+nDotIndex, imgSiftSaveAddress.begin()+nLength-1);
                    imgSiftSaveAddress.append("_sift_bow");
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

                    std::string imgTextonSaveAddress = imgAddress;
                    imgTextonSaveAddress.erase(imgTextonSaveAddress.begin()+ nDotIndex, imgTextonSaveAddress.begin()+nLength-1);
                    imgTextonSaveAddress.append("_texton_bow");
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

                if(param.useTextonIFV){
                    cv::Mat textonIfv;
                    extractTextonIFV(img, mask, textonIfv, m_textonGMMDist);

                    std::string imgTextonIfvSaveAddress = imgAddress;
                    imgTextonIfvSaveAddress.erase(imgTextonIfvSaveAddress.begin()+nDotIndex, imgTextonIfvSaveAddress.begin()+nLength-1);
                    imgTextonIfvSaveAddress.append("_texton_ifv");
                    Miscellaneous::IO::data2Text_<float>(textonIfv, imgTextonIfvSaveAddress);

                    float *srcPtr;
                    float *dstPtr ;
                    if(nImgCnt >= nTrainDataSize){
                        srcPtr = textonIfv.ptr<float>(0);
                        dstPtr = textonIfvTestDataSet.ptr<float>(nTestDataCnt);
                        for(int k = 0; k < ms_nTextonIFVDimension; ++k){
                            *dstPtr = *srcPtr;
                            ++dstPtr;
                            ++srcPtr;
                        }
                    }else{
                        srcPtr = textonIfv.ptr<float>(0);
                        dstPtr = textonIfvTrainDataSet.ptr<float>(nTrainDataCnt);
                        for(int k = 0; k < ms_nTextonIFVDimension; ++k){
                            *dstPtr = *srcPtr;
                            ++dstPtr;
                            ++srcPtr;
                        }
                    }
                }

                if(param.useSiftIFV){
                    cv::Mat siftIfv;
                    extractSiftIFV(img, mask, siftIfv, m_siftGMMDist);

                    std::string imgSiftIfvAddress = imgAddress;
                    imgSiftIfvAddress.erase(imgSiftIfvAddress.begin()+ nDotIndex, imgSiftIfvAddress.begin()+nLength-1);
                    imgSiftIfvAddress.append("_sift_ifv");
                    Miscellaneous::IO::data2Text_<float>(siftIfv, imgSiftIfvAddress);

                    float *srcPtr;
                    float *dstPtr ;
                    if(nImgCnt >= nTrainDataSize){
                        srcPtr = siftIfv.ptr<float>(0);
                        dstPtr = siftIfvTestDataSet.ptr<float>(nTestDataCnt);
                        for(int k = 0; k < ms_nSiftIFVDimension; ++k){
                            *dstPtr = *srcPtr;
                            ++dstPtr;
                            ++srcPtr;
                        }
                    }else{
                        srcPtr = siftIfv.ptr<float>(0);
                        dstPtr = siftIfvTrainDataSet.ptr<float>(nTrainDataCnt);
                        for(int k = 0; k < ms_nSiftIFVDimension; ++k){
                            *dstPtr = *srcPtr;
                            ++dstPtr;
                            ++srcPtr;
                        }
                    }
                }

                if(nImgCnt >= nTrainDataSize){
                    labelTestDataSet.at<float>(nTestDataCnt, 0) = i;
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

            if(param.useColorIFV){
                colorBoWSet.adjustROI(0, nMaxSingleClassSize-nImgCnt, 0, 0);
                vecColorBoWSet.push_back(colorBoWSet);
            }

            labelSet.adjustROI(0, nMaxSingleClassSize-nImgCnt, 0, 0);
            vecLabelSet.push_back(labelSet);

            closedir(pDir);
        }
        labelTrainDataSet.adjustROI(0,  nTrainDataCnt-nMaxDataSize, 0, 0);
        siftBowTrainDataSet.adjustROI(0, nTrainDataCnt-nMaxDataSize, 0, 0);
        textonBowTrainDataSet.adjustROI(0, nTrainDataCnt-nMaxDataSize, 0, 0);
        colorIfvTrainDataSet.adjustROI(0, nTrainDataCnt-nMaxDataSize, 0, 0);
        siftIfvTrainDataSet.adjustROI(0, nTrainDataCnt-nMaxDataSize, 0, 0);
        textonIfvTrainDataSet.adjustROI(0, nTrainDataCnt-nMaxDataSize, 0, 0);

        labelTestDataSet.adjustROI(0, nTestDataCnt-nMaxDataSize, 0, 0);
        siftBowTestDataSet.adjustROI(0, nTestDataCnt-nMaxDataSize, 0, 0);
        textonBowTestDataSet.adjustROI(0, nTestDataCnt-nMaxDataSize, 0, 0);
        colorIfvTestDataSet.adjustROI(0, nTestDataCnt-nMaxDataSize, 0, 0);
        siftIfvTestDataSet.adjustROI(0, nTestDataCnt-nMaxDataSize, 0, 0);
        textonIfvTestDataSet.adjustROI(0, nTrainDataCnt-nMaxDataSize, 0, 0);

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

        if(param.useSiftIFV){
            Miscellaneous::IO::data2Text_<float>(siftIfvTrainDataSet, dataAddress+"/org_sift_ifv_train_data");
            Miscellaneous::IO::data2Text_<float>(siftIfvTestDataSet, dataAddress+"/org_sift_ifv_test_data");
        }

        if(param.useColorIFV){
            Miscellaneous::IO::data2Text_<float>(colorIfvTrainDataSet, dataAddress+"/org_color_ifv_train_data");
            Miscellaneous::IO::data2Text_<float>(colorIfvTestDataSet, dataAddress+"/org_color_ifv_test_data");
        }

        if(param.useTextonIFV){
            Miscellaneous::IO::data2Text_<float>(textonIfvTrainDataSet, dataAddress+"/org_texton_ifv_train_data");
            Miscellaneous::IO::data2Text_<float>(textonIfvTestDataSet, dataAddress+"/org_texton_ifv_test_data");
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

        if(param.useTextonIFV){
            //            Miscellaneous::IO::readData_<float>(textonIfvTrainDataSet, dataAddress+"/org_texton_ifv_train_data");
            //            Miscellaneous::IO::readData_<float>(textonIfvTestDataSet, dataAddress+"/org_texton_ifv_test_data");
            Miscellaneous::IO::readData_<float>(textonIfvTrainDataSet, "/home/shenyunjun/Data/new_data/fmd/org_texton_ifv_train_data");
            Miscellaneous::IO::readData_<float>(textonIfvTestDataSet, "/home/shenyunjun/Data/new_data/fmd/org_texton_ifv_test_data");
        }

        if(param.useSiftIFV){
//            Miscellaneous::IO::readData_<float>(siftIfvTrainDataSet, dataAddress+"/org_sift_ifv_train_data");
//            Miscellaneous::IO::readData_<float>(siftIfvTestDataSet, dataAddress+"/org_sift_ifv_test_data");

                         cv::Mat normTrainDataSet, normTestDataSet, microTrainDataSet, microTestDataSet;
                        Miscellaneous::IO::readData_<float>(normTrainDataSet, dataAddress+"/org_sift_ifv_train_data");
                        Miscellaneous::IO::readData_<float>(normTestDataSet, dataAddress+"/org_sift_ifv_test_data");
                        Miscellaneous::IO::readData_<float>(microTrainDataSet, "/home/shenyunjun/Data/new_data/fmd/org_sift_ifv_train_data");
                        Miscellaneous::IO::readData_<float>(microTestDataSet, "/home/shenyunjun/Data/new_data/fmd/org_sift_ifv_test_data");

                        int normLength = normTestDataSet.cols;
                        int microLength = microTestDataSet.cols;

                        siftIfvTrainDataSet = cv::Mat_<float>(normTrainDataSet.rows, normLength+microLength);
                        siftIfvTestDataSet = cv::Mat_<float>(normTestDataSet.rows, normLength+microLength);

                        siftIfvTestDataSet.adjustROI(0,0,0, -microLength);
                        siftIfvTrainDataSet.adjustROI(0, 0, 0, -microLength);
                        normTrainDataSet.copyTo(siftIfvTrainDataSet);
                        normTestDataSet.copyTo(siftIfvTestDataSet);
                        siftIfvTestDataSet.adjustROI(0,0, -normLength, microLength);
                        siftIfvTrainDataSet.adjustROI(0, 0, -normLength, microLength);
                        microTrainDataSet.copyTo(siftIfvTrainDataSet);
                        microTestDataSet.copyTo(siftIfvTestDataSet);
                        siftIfvTestDataSet.adjustROI(0,0, normLength, 0);
                        siftIfvTrainDataSet.adjustROI(0, 0, normLength, 0);
        }

        if(param.useColorIFV){
            Miscellaneous::IO::readData_<float>(colorIfvTrainDataSet, dataAddress+"/org_color_ifv_train_data");
            Miscellaneous::IO::readData_<float>(colorIfvTestDataSet, dataAddress+"/org_color_ifv_test_data");
        }
    }


    //combine features to do lda
    //    cv::LDA lda;
    //    cv::Mat projectedsiftBowTrainDataSet;
    //    cv::Mat projectedtextonBowTrainDataSet;
    //    cv::Mat projectedcolorBowTrainDataSet;
    //    cv::Mat projectedSiftIfvTrainDataSet;

    //    cv::PCA pca;
    //    //pca
    //    if(param.useTexton){

    //    }

    //    if(param.useSIFT){
    //        lda.compute(siftBowTrainDataSet, labelTrainDataSet);
    //        cv::Mat siftEigenVector = lda.eigenvectors();
    //        siftEigenVector.convertTo(siftEigenVector, CV_32F);
    //        Miscellaneous::IO::data2Text_<float>(siftEigenVector,dataAddress+"/sift_eigen");
    //        projectedsiftBowTrainDataSet = siftBowTrainDataSet*siftEigenVector;
    //    }

    //    if(param.useTexton){
    //        lda.compute(textonBowTrainDataSet, labelTrainDataSet);
    //        cv::Mat textonEigenVector = lda.eigenvectors();
    //        textonEigenVector.convertTo(textonEigenVector, CV_32F);
    //        Miscellaneous::IO::data2Text_<float>(textonEigenVector, dataAddress+"/texton_eigen");
    //        projectedtextonBowTrainDataSet = textonBowTrainDataSet*textonEigenVector;
    //    }

    cv::Mat trainData, testData, trainLabel, testLabel;

    labelTrainDataSet.convertTo(trainLabel, CV_32S);
    labelTestDataSet.convertTo(testLabel, CV_32S);

    if(param.useColorIFV){
        int nCurCol = trainData.cols;
        int nExtraCol = ms_nColorIFVDimension;
        int nTrainDataNum = colorIfvTrainDataSet.rows;
        int nTestDataNum = colorIfvTestDataSet.rows;
        cv::Mat extraTrainData = colorIfvTrainDataSet;
        cv::Mat extraTestData = colorIfvTestDataSet;

        if(trainData.empty()){
            trainData = extraTrainData;
            testData = extraTestData;
        }else{
            cv::Mat newTrainData = cv::Mat_<float>(nTrainDataNum, nCurCol+nExtraCol);
            cv::Mat newTestData = cv::Mat_<float>(nTestDataNum, nCurCol+nExtraCol);

            newTrainData.adjustROI(0,0,0,-nExtraCol);
            trainData.copyTo(newTrainData);
            newTrainData.adjustROI(0,0,-nCurCol,nExtraCol);
            extraTrainData.copyTo(newTrainData);
            newTrainData.adjustROI(0, 0, nCurCol, 0);

            newTestData.adjustROI(0,0,0,-nExtraCol);
            testData.copyTo(newTestData);
            newTestData.adjustROI(0,0,-nCurCol,nExtraCol);
            extraTestData.copyTo(newTestData);
            newTestData.adjustROI(0, 0, nCurCol, 0);

            trainData = newTrainData;
            testData = newTestData;
        }
    }

    if(param.useTexton){
        int nCurCol = trainData.cols;
        int nExtraCol = ms_nGlobalTextonDictionarySize;
        int nTrainDataNum = textonBowTrainDataSet.rows;
        int nTestDataNum = textonBowTestDataSet.rows;
        cv::Mat extraTrainData = textonBowTrainDataSet;
        cv::Mat extraTestData = textonBowTestDataSet;

        if(trainData.empty()){
            trainData = extraTrainData;
            testData = extraTestData;
        }else{
            cv::Mat newTrainData = cv::Mat_<float>(nTrainDataNum, nCurCol+nExtraCol);
            cv::Mat newTestData = cv::Mat_<float>(nTestDataNum, nCurCol+nExtraCol);

            newTrainData.adjustROI(0,0,0,-nExtraCol);
            trainData.copyTo(newTrainData);
            newTrainData.adjustROI(0,0,-nCurCol,nExtraCol);
            extraTrainData.copyTo(newTrainData);
            newTrainData.adjustROI(0, 0, nCurCol, 0);

            newTestData.adjustROI(0,0,0,-nExtraCol);
            testData.copyTo(newTestData);
            newTestData.adjustROI(0,0,-nCurCol,nExtraCol);
            extraTestData.copyTo(newTestData);
            newTestData.adjustROI(0, 0, nCurCol, 0);

            trainData = newTrainData;
            testData = newTestData;
        }
    }

    if(param.useTextonIFV){
        int nCurCol = trainData.cols;
        int nExtraCol = ms_nTextonIFVDimension;
        int nTrainDataNum = textonIfvTrainDataSet.rows;
        int nTestDataNum = textonIfvTestDataSet.rows;
        cv::Mat extraTrainData = textonIfvTrainDataSet;
        cv::Mat extraTestData = textonIfvTestDataSet;

        if(trainData.empty()){
            trainData = extraTrainData;
            testData = extraTestData;
        }else{
            cv::Mat newTrainData = cv::Mat_<float>(nTrainDataNum, nCurCol+nExtraCol);
            cv::Mat newTestData = cv::Mat_<float>(nTestDataNum, nCurCol+nExtraCol);

            newTrainData.adjustROI(0,0,0,-nExtraCol);
            trainData.copyTo(newTrainData);
            newTrainData.adjustROI(0,0,-nCurCol,nExtraCol);
            extraTrainData.copyTo(newTrainData);
            newTrainData.adjustROI(0, 0, nCurCol, 0);

            newTestData.adjustROI(0,0,0,-nExtraCol);
            testData.copyTo(newTestData);
            newTestData.adjustROI(0,0,-nCurCol,nExtraCol);
            extraTestData.copyTo(newTestData);
            newTestData.adjustROI(0, 0, nCurCol, 0);

            trainData = newTrainData;
            testData = newTestData;
        }
    }



    if(param.useSiftIFV){
        int nCurCol = trainData.cols;
        int nExtraCol = ms_nSiftIFVDimension;
        int nTrainDataNum = siftIfvTrainDataSet.rows;
        int nTestDataNum = siftIfvTestDataSet.rows;
        cv::Mat extraTrainData = siftIfvTrainDataSet;
        cv::Mat extraTestData = siftIfvTestDataSet;

        if(trainData.empty()){
            trainData = extraTrainData;
            testData = extraTestData;
        }else{
            cv::Mat newTrainData = cv::Mat_<float>(nTrainDataNum, nCurCol+nExtraCol);
            cv::Mat newTestData = cv::Mat_<float>(nTestDataNum, nCurCol+nExtraCol);

            newTrainData.adjustROI(0,0,0,-nExtraCol);
            trainData.copyTo(newTrainData);
            newTrainData.adjustROI(0,0,-nCurCol,nExtraCol);
            extraTrainData.copyTo(newTrainData);
            newTrainData.adjustROI(0, 0, nCurCol, 0);

            newTestData.adjustROI(0,0,0,-nExtraCol);
            testData.copyTo(newTestData);
            newTestData.adjustROI(0,0,-nCurCol,nExtraCol);
            extraTestData.copyTo(newTestData);
            newTestData.adjustROI(0, 0, nCurCol, 0);

            trainData = newTrainData;
            testData = newTestData;
        }
    }



    //    int nDataDim = textonBowTrainDataSet.cols;
    //    int nDataSize = textonBowTrainDataSet.rows;


    //    double trainData[nDataSize*nDataDim];
    //    double trainLabel[nDataSize];

    //    float testData[nDataSize*nDataDim];
    //    float testLabel[nDataSize];


    //    vl_size const numData = nDataSize ;
    //    vl_size const dimension = nDataDim ;

    //    for(int i = 0 ; i <nDataSize; ++i){
    //        for(int j = 0; j < nDataDim; ++j){
    //            trainData[i*nDataDim + j] = textonBowTrainDataSet.at<float>(i,j);
    //        }
    //        trainLabel[i] = labelTrainDataSet.at<float>(i,0);
    //    }

    //    std::vector<cv::Mat> modelSet;
    //    std::vector<float> biasSet;

    //    double lambda = 0.01;

    //    cv::Mat combined = cv::Mat_<float>(projectedtextonBowTrainDataSet.rows, projectedtextonBowTrainDataSet.cols*2);
    //    combined.adjustROI(0, 0, 0, -9);
    //    projectedtextonBowTrainDataSet.copyTo(combined);
    //    combined.adjustROI(0, 0, -9, 9);
    //    projectedsiftBowTrainDataSet.copyTo(combined);
    //    combined.adjustROI(0, 0 , 9, 0);


    cv::ml::KNearest::Params knnParam;
    knnParam.isclassifier = true;
    knnParam.defaultK = 30;
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::StatModel::train<cv::ml::KNearest>(trainData, cv::ml::ROW_SAMPLE, trainLabel, knnParam);

    cv::ml::SVM::Params svmParams;
    double epsilon = 1e-7;
    svmParams.svmType = cv::ml::SVM::C_SVC;
    svmParams.kernelType = cv::ml::SVM::LINEAR;
    svmParams.degree = 3;
    svmParams.gamma = 1;
    svmParams.C = 100;
    svmParams.nu = 0.1;
    svmParams.termCrit = cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 100000, epsilon);
    {
        cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::train<cv::ml::SVM>(trainData, cv::ml::ROW_SAMPLE, trainLabel, svmParams);

        // compute E_in
        cv::Mat predictedTrainLabel;
        svm->predict(trainData, predictedTrainLabel);
        //        knn->findNearest(trainData, 30, predictedTrainLabel);
        double Ein = 0;
        cv::Mat confuseMat = cv::Mat_<float>(10, 10, float(0));
        for(int i = 0; i < predictedTrainLabel.rows; ++i){
            int gndTruth = trainLabel.at<int>(i,0);
            int predict = (int)predictedTrainLabel.at<float>(i,0);
            confuseMat.at<float>(gndTruth, predict) = confuseMat.at<float>(gndTruth, predict)+1;
            if(predict != gndTruth){
                Ein += 1;
            }
        }

        Ein = Ein/predictedTrainLabel.rows;

        std::cout<<"Trained Ein is "<<Ein<<std::endl;
        std::cout<<"confuse mat\n";
        std::cout<<(confuseMat/50.0)<<std::endl;

        std::cout.flush();

        //compute E_out
        cv::Mat predictedTestLabel;
        svm->predict(testData, predictedTestLabel);

        //        knn->findNearest(testData, 30, predictedTestLabel);
        double Eout = 0;
        confuseMat = cv::Mat_<float>(10, 10, float(0));
        for(int i = 0; i < predictedTestLabel.rows; ++i){

            int gndTruth = testLabel.at<int>(i,0);
            int predict = (int)predictedTestLabel.at<float>(i,0);
            //            std::cout<<gndTruth<<" "<<predict<<";"<<std::endl;
            confuseMat.at<float>(gndTruth, predict) = confuseMat.at<float>(gndTruth, predict)+1;
            if(predict != gndTruth){
                Eout += 1;
            }
        }
        for(int i = 0; i < confuseMat.rows; ++i){
            std::cout<<confuseMat.at<float>(i,i)/50.0<<';'<<std::endl;
        }

        Eout = Eout/predictedTestLabel.rows;

        std::cout<<"Trained Eout is "<<Eout<<std::endl;
        std::cout<<"confuse mat\n"<<(confuseMat/50.0)<<std::endl;
        std::cout.flush();
        cv::namedWindow("confuseMat", cv::WINDOW_NORMAL);
        imshow("confuseMat", confuseMat/50.0);
        cv::waitKey(0);

    }
    //    for(int i = 0; i < 10; ++i){
    //        double singleLabelArray[label.rows];
    //        for(int j = 0; j < label.rows; ++j){
    //            if(label.at<int>(j,0) == i){
    //                singleLabelArray[j] = 1;
    //            }else{
    //                singleLabelArray[j] = -1;
    //            }
    //        }

    //        const double * model ;
    //        double bias ;

    //        VlSvm * vlsvm = vl_svm_new(VlSvmSolverSgd,
    //                                   trainData, dimension, numData,
    //                                   singleLabelArray,
    //                                   lambda) ;

    //        vl_svm_train(vlsvm);

    //        bias = vl_svm_get_bias(vlsvm);
    //        model = vl_svm_get_model(vlsvm);
    //        int length = vl_svm_get_num_data(vlsvm);

    //        cv::Mat modelMat = cv::Mat_<float>(dimension,1);
    //        for(int l = 0; l < dimension; ++l){
    //            modelMat.at<float>(l, 0) = static_cast<float>(model[l]);
    //        }

    //        modelSet.push_back(modelMat);

    //        biasSet.push_back(bias);
    //    }

    //    double E_in = 0, E_out = 0;
    //    //compute E_in
    //    std::cout<<"E_in predicted\n";
    //    for(int i = 0; i < labelTrainDataSet.rows; ++i){
    //        double maxMarg = 0;
    //        int maxIndex = 0;
    //        for(int j = 0; j < 10; ++j){
    //            cv::Mat dotPro = textonBowTrainDataSet.row(i)* modelSet[j];
    //            double marg = dotPro.at<float>(0,0) + biasSet[j];
    //            if(std::fabs(marg) > maxMarg){
    //                maxIndex = j;
    //                maxMarg = std::fabs(marg);
    //            }
    //        }
    //        float diff = (labelTrainDataSet.at<float>(i,0) - maxIndex);
    //        std::cout<<maxIndex<<"  "<<labelTrainDataSet.at<float>(i,0)<<" diff:"<<diff<<std::endl;

    //        if(std::fabs(diff) > 0.5){
    //            E_in += 1;
    //        }
    //    }

    ////    E_in = E_in/labelTrainDataSet.rows;
    //    std::cout<<"total data size"<<labelTrainDataSet.rows<<std::endl;

    //    std::cout<<"E_in "<<E_in<<std::endl;
    //    std::cout.flush();

    //    //compute E_out
    //    for(int i = 0; i < labelTestDataSet.rows; ++i){
    //        double maxMarg = 0;
    //        int maxIndex = 0;
    //        for(int j = 0; j < 10; ++j){
    //            cv::Mat dotPro = textonBowTestDataSet.row(i)* modelSet[j];
    //            double marg = dotPro.at<float>(0,0) + biasSet[j];
    //            if(std::fabs(marg) > maxMarg){
    //                maxIndex = j;
    //                maxMarg = std::fabs(marg);
    //            }
    //        }
    //        if((labelTestDataSet.at<float>(i,0) - maxIndex) > 0.5){
    //            E_out += 1;
    //        }
    //    }

    //    E_out = E_out/labelTrainDataSet.rows;

    //    std::cout<<"E_out "<<E_out<<std::endl;
    //    std::cout.flush();



    //    // compute E_in
    //    cv::Mat predictedTrainLabel;
    //    svm->predict(trainData, predictedTrainLabel);
    //    //    knn->findNearest(trainData, 30, predictedTrainLabel);

    //    double Ein = 0;
    //    for(int i = 0; i < predictedTrainLabel.rows; ++i){
    //        //        std::cout<<"predicted: "<<predictedTrainLabel.at<float>(i,0)<<" gt: "<<labelTrainDataSet.at<float>(i, 0)<<std::endl;
    //        if(std::fabs(predictedTrainLabel.at<float>(i,0) - trainLabel.at<int>(i,0)) > 0.5){
    //            Ein += 1;
    //        }
    //    }
    //    Ein = Ein/predictedTrainLabel.rows;

    //    std::cout<<"Trained Ein is "<<Ein<<std::endl;
    //    std::cout.flush();

    //    //compute E_out
    //    cv::Mat predictedTestLabel;
    //    svm->predict(testData, predictedTestLabel);
    //    //    knn->findNearest(testData, 30, predictedTestLabel);

    //    std::cout<<"Eout result:\n";
    //    double Eout = 0;
    //    for(int i = 0; i < predictedTestLabel.rows; ++i){
    //        //        std::cout<<"predicted: "<<predictedTestLabel.at<float>(i,0)<<" gt: "<<labelTestDataSet.at<float>(i, 0)<<std::endl;
    //        if(std::fabs(predictedTestLabel.at<float>(i,0) - testLabel.at<int>(i,0)) > 0.5){
    //            Eout += 1;
    //        }
    //    }
    //    Eout = Eout/predictedTestLabel.rows;

    //    std::cout<<"Trained Eout is "<<Eout<<std::endl;
    //    std::cout.flush();

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

void MaterialClassifier::extractTextonDist(cv::Mat img, cv::Mat &mask,  cv::Mat &textonDist)
{
    cv::Mat f_grayImg, grayImg;
    if(img.channels() == 1){
        img.convertTo(f_grayImg, CV_32F);
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

    //    visualize response map
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
    cv::Mat responseVec = cv::Mat_<float>(totalResponseMat.rows*totalResponseMat.cols, totalResponseMat.channels());
    for(int j = 0; j < totalResponseMat.rows; ++j){
        float *srcPtr = totalResponseMat.ptr<float>(j);
        for(int k = 0; k < totalResponseMat.cols; ++k){
            float *dstPtr = responseVec.ptr<float>(j*nCol+k);
            for(int l = 0; l < totalResponseMat.channels(); ++l){
                *dstPtr++ = *srcPtr++;
            }
        }
    }


    float voteBins[ms_nGlobalTextonDictionarySize];
    double voteSum = 0.01;
    std::fill(voteBins, voteBins+ms_nGlobalTextonDictionarySize, float(0));


    int nSampleStep = 5;
    for(int i = 0; i < nRow; i+= nSampleStep ){
        uchar *maskPtr = mask.ptr<uchar>(i);
        for(int k = 0; k < nCol; k+=nSampleStep){
            int maskValue = (*maskPtr);
            maskPtr = maskPtr + nSampleStep;
            if(maskValue < 10){
                continue;
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

            float weightDist[ms_nSiftDictionarySize];
            float weightSum = 0.01;
            for(int j = 0; j < ms_nGlobalTextonDictionarySize; ++j){
                cv::Mat diff = testVector - m_globalTextonDictionary.row(j);
                double dotProduct = diff.dot(diff);
                double prob = std::exp(-dotProduct);
                weightSum = weightSum + prob;
                weightDist[j] = prob;
            }

            for(int j = 0; j < ms_nGlobalTextonDictionarySize; ++j){
                voteBins[j] += weightDist[j]/weightSum;
            }

            voteSum += 1;
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

void MaterialClassifier::extractTextonIFV(Mat img, Mat mask, Mat &textonIfv, GMM &gmmDist)
{
    cv::Mat f_grayImg, grayImg;
    if(img.channels() == 1){
        img.convertTo(f_grayImg, CV_32F);
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

//    //     visualize response map
//    for(int i = 0; i < responseMap.size(); ++i){
//        QString fileName = QString::number(i)+"_response.tiff";
//        cv::Mat visualImg = responseMap[i].clone();
//        visualImg = cv::abs(visualImg);
//        double min,max;
//        cv::minMaxLoc(visualImg,&min, &max);
//        cv::Mat saveImg;
//        visualImg = 255*(visualImg - min)/(max-min);
//        visualImg.convertTo(saveImg,CV_8U);
//        cv::imwrite(fileName.toLocal8Bit().data(), saveImg);
//    }
//    cv::Mat responseVec = cv::Mat_<float>(totalResponseMat.rows*totalResponseMat.cols, totalResponseMat.channels());
//    for(int j = 0; j < totalResponseMat.rows; ++j){
//        float *srcPtr = totalResponseMat.ptr<float>(j);
//        for(int k = 0; k < totalResponseMat.cols; ++k){
//            float *dstPtr = responseVec.ptr<float>(j*nCol+k);
//            for(int l = 0; l < totalResponseMat.channels(); ++l){
//                *dstPtr++ = *srcPtr++;
//            }
//        }
//    }


    int nSampleStep = 5;
    float filterRespBuffer[m_nFilterKernelSetSize*nCol*nRow/(nSampleStep*nSampleStep-1)];


    int nDataCnt = 0;
    for(int i = 0; i < nRow; i+= nSampleStep ){
        uchar *maskPtr = mask.ptr<uchar>(i);
        for(int k = 0; k < nCol; k+=nSampleStep){
            int maskValue = (*maskPtr);
            maskPtr = maskPtr + nSampleStep;
            if(maskValue < 10){
                continue;
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

            float *dstPtr = filterRespBuffer+nDataCnt*m_nFilterKernelSetSize;
            for(int j = 0; j < m_nFilterKernelSetSize; ++j){
                dstPtr[j] = responsePtr[j];
            }
            ++nDataCnt;
        }
    }

    int ifvLength = 2 * gmmDist.dimension * gmmDist.clusterNum;

    // allocate space for the encoding
    void *enc = vl_malloc(sizeof(float) * ifvLength);

    // run fisher encoding
    vl_fisher_encode(
                enc,
                VL_TYPE_FLOAT,
                gmmDist.mean,
                gmmDist.dimension,
                gmmDist.clusterNum,
                gmmDist.covariance,
                gmmDist.prior,
                filterRespBuffer, nDataCnt,
                VL_FISHER_FLAG_IMPROVED
                ) ;

    cv::Mat textonHist = cv::Mat_<float>(1, ifvLength);

    float *ifvDstPtr = textonHist.ptr<float>(0);
    float *ifvSrcPtr = static_cast<float*>(enc);
    for(int i = 0; i < ifvLength; ++i){
        ifvDstPtr[i] = ifvSrcPtr[i];
    }

    textonIfv = textonHist;
}

void MaterialClassifier::extractColorIFV(Mat img, Mat mask, Mat &colorIfv, GMM &gmmDist)
{
    //sample response value
    int nSampleStep = 5;
    int nColorPatchLength = 27;

    int nColorPatchSize = 9;
    int nHalfColorPatchWidth = 1;
    int shiftX[9], shiftY[9];
    {
        int cnt = 0;
        for(int i = -1; i < 2; ++i){
            for(int j = -1; j < 2; ++j){
                shiftX[cnt] = i;
                shiftY[cnt] = j;
                ++cnt;
            }

        }
    }

    int nCol = img.cols;
    int nRow = img.rows;
    int nMaxX = nCol - nHalfColorPatchWidth;
    int nMinX = nHalfColorPatchWidth;
    int nMaxY = nRow - nHalfColorPatchWidth;
    int nMinY = nHalfColorPatchWidth;

    float colorPatchBuffer[nColorPatchLength*nCol*nRow/(nSampleStep*nSampleStep-1)];
    int nDataCnt = 0;
    for(int y = nMinY; y < nMaxY; y+=nSampleStep){
        for(int x = nMinX; x < nMaxX; x+=nSampleStep){
            uchar maskValue = mask.at<uchar>(y, x);
            if(maskValue <= 1){
                continue;
            }


            float *dstPtr = colorPatchBuffer + nColorPatchLength*nDataCnt;

            for(int j = 0; j < nColorPatchSize; ++j){
                cv::Vec3b bgr = img.at<cv::Vec3b>(y+shiftY[j], x+shiftX[j]);
                *dstPtr = bgr[0]; ++dstPtr;
                *dstPtr = bgr[1]; ++dstPtr;
                *dstPtr = bgr[2]; ++dstPtr;

            }

            ++nDataCnt;
        }
    }

    int ifvLength = 2 * gmmDist.dimension * gmmDist.clusterNum;

    // allocate space for the encoding
    void *enc = vl_malloc(sizeof(float) * ifvLength);

    // run fisher encoding
    vl_fisher_encode(
                enc,
                VL_TYPE_FLOAT,
                gmmDist.mean,
                gmmDist.dimension,
                gmmDist.clusterNum,
                gmmDist.covariance,
                gmmDist.prior,
                colorPatchBuffer, nDataCnt,
                VL_FISHER_FLAG_IMPROVED
                ) ;

    cv::Mat colorPatchIfv = cv::Mat_<float>(1, ifvLength);

    float *ifvDstPtr = colorPatchIfv.ptr<float>(0);
    float *ifvSrcPtr = static_cast<float*>(enc);
    for(int i = 0; i < ifvLength; ++i){
        ifvDstPtr[i] = ifvSrcPtr[i];
    }

    colorIfv = colorPatchIfv;


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
    VlDsiftFilter *dsift = vl_dsift_new_basic(nCol, nRow, 8, 12);
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
        float weightDist[ms_nSiftDictionarySize];
        float weightSum = 0;
        for(int j = 0; j < ms_nSiftDictionarySize; ++j){
            cv::Mat diff = siftDescriptor - m_siftDictionary.row(j);
            double prob = std::exp(-diff.dot(diff));
            weightDist[j] = prob;
            weightSum = weightSum + prob;
        }

        //voting
        for(int j = 0 ; j < ms_nSiftDictionarySize; ++j){
            votedBin[j] += weightDist[j]/weightSum;
        }
        voteSum = voteSum + 1;
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

void MaterialClassifier::extractSiftIFV(Mat _img, Mat mask, Mat &siftIfv, GMM &gmmDist)
{
    int nOrgRow = _img.rows;
    int nOrgCol = _img.cols;

    float siftDescriptorBuffer[128*10000];

    cv::Mat img;
    if(_img.channels() == 3){
        cv::cvtColor(_img, img, cv::COLOR_BGR2GRAY);
    }else{
        img = _img;
    }

    //detect dense sift
    int nDspCnt = 0;
    for(int i = 0; i < 3; ++i){
        int nRow = nOrgRow*m_siftScale[i];
        int nCol = nOrgCol*m_siftScale[i];

        cv::Mat scaledImg, scaledMask;
        cv::resize(img, scaledImg, cv::Size(nCol, nRow));
        cv::resize(mask, scaledMask, cv::Size(nCol, nRow));
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
        VlDsiftFilter *dsift = vl_dsift_new_basic(nCol, nRow, 6, m_nSiftBinNum);
        vl_dsift_process(dsift, imgArray);
        const float *siftDescriptorPtr = vl_dsift_get_descriptors(dsift);
        int nDescriptorSize = vl_dsift_get_descriptor_size(dsift);
        int nKeyPointNum = vl_dsift_get_keypoint_num(dsift);
        const VlDsiftKeypoint *keyPoints =  vl_dsift_get_keypoints(dsift);


        for(int j = 0; j < nKeyPointNum; ++j){
            int x = keyPoints[j].x;
            int y = keyPoints[j].y;
            if(scaledMask.at<uchar>(y,x) <  200){
                continue;
            }
            float *dstPtr = siftDescriptorBuffer + nDspCnt*nDescriptorSize;
            const float *srcPtr = siftDescriptorPtr + j*nDescriptorSize;
            float l1Norm = 0.01;
            for(int k = 0; k < nDescriptorSize; ++k){
                dstPtr[k] = srcPtr[k];
                if(srcPtr[k] > 0){
                    l1Norm = l1Norm + srcPtr[k];
                }else{
                    l1Norm = l1Norm - srcPtr[k];
                }
            }

            dstPtr = siftDescriptorBuffer + nDspCnt*nDescriptorSize;
            for(int k = 0; k < nDescriptorSize; ++k){
                dstPtr[k] = dstPtr[k]/l1Norm;
            }

            ++nDspCnt;
        }

        vl_dsift_delete(dsift);
    }
    int ifvLength = 2 * gmmDist.dimension * gmmDist.clusterNum;

    // allocate space for the encoding
    void *enc = vl_malloc(sizeof(float) * ifvLength);

    // run fisher encoding
    vl_fisher_encode(
                enc,
                VL_TYPE_FLOAT,
                gmmDist.mean,
                gmmDist.dimension,
                gmmDist.clusterNum,
                gmmDist.covariance,
                gmmDist.prior,
                siftDescriptorBuffer, nDspCnt,
                VL_FISHER_FLAG_IMPROVED
                ) ;

    cv::Mat siftHist = cv::Mat_<float>(1, ifvLength);

    float *ifvDstPtr = siftHist.ptr<float>(0);
    float *ifvSrcPtr = static_cast<float*>(enc);
    for(int i = 0; i < ifvLength; ++i){
        ifvDstPtr[i] = ifvSrcPtr[i];
    }

    siftIfv = siftHist;
}

void MaterialClassifier::buildFilterKernelSet(string dataAddress)
{

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

    int nMaxDataSize = 800000;
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

            if(mask.empty()){
                mask = cv::Mat_<uchar>(img.rows, img.cols, (uchar)255);
            }

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
        cv::imwrite(QString(name+".tiff").toLocal8Bit().data(), kernelImg);
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

void MaterialClassifier::buildTextonGmm(string dataAddress)
{
    std::cout<<"start build texton dictionary\n";
    std::cout.flush();

    int nHalfKernelSize = m_nFilterKernelWidth/2;

    int nMaxDataSize = 800000;
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
            if(nImgCnt > 50){
                continue;
            }
            ++nImgCnt;

            if(mask.empty()){
                mask = cv::Mat_<uchar>(img.rows, img.cols, (uchar)255);
            }

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


    filterResponseSet.adjustROI(0,nDataCnt-nMaxDataSize,0,0);

    int descriptorNum = filterResponseSet.rows;
    int descriptorSize = filterResponseSet.cols;

    int maxArraySize = std::min(descriptorNum*descriptorSize, 1000000);
    float data[maxArraySize];

    int maxNum = maxArraySize/descriptorSize;
    for(int i = 0; i < maxNum; ++i){
        for(int j = 0; j <descriptorSize; ++j){
            data[i*descriptorSize + j] = filterResponseSet.at<float>(i,j);
        }
    }

    int dimension = descriptorSize;
    int numData = maxNum;
    int numClusters = ms_nTextonGMMClusterNum;

    VlGMM *gmm = vl_gmm_new (VL_TYPE_FLOAT, dimension, numClusters) ;
    vl_gmm_cluster(gmm, data, numData);

    const void *gmmMeans = vl_gmm_get_means(gmm);
    const void *gmmCovariance = vl_gmm_get_covariances(gmm);
    const void *prior = vl_gmm_get_priors(gmm);

    cv::Mat textonGmmInfo = cv::Mat_<float>(numClusters, dimension+dimension+1);//mean + digonal covariance + prior

    for(int i = 0; i < numClusters; ++i){
        float *dstGmmPtr = textonGmmInfo.ptr<float>(i);
        const float *fGmmMeans = static_cast<const float*>(gmmMeans);
        const float *fGmmCov = static_cast<const float*>(gmmCovariance);
        const float *fGmmPrior = static_cast<const float*>(prior);

        for(int j = 0; j < dimension; ++j){
            *dstGmmPtr = fGmmMeans[i*dimension + j];
            ++dstGmmPtr;
        }

        for(int j = 0; j < dimension; ++j){
            *dstGmmPtr = fGmmCov[i*dimension + j];
            ++dstGmmPtr;
        }

        *dstGmmPtr = fGmmPrior[i];
    }

    Miscellaneous::IO::data2Text_<float>(textonGmmInfo, dataAddress+"/texton_gmm_info");

    m_textonGMMDist.load(std::string(dataAddress+"/texton_gmm_info"), dimension);
}

void MaterialClassifier::buildGlobalTextonDictionary(string dataAddress)
{
    std::cout<<"start build texton dictionary\n";
    std::cout.flush();

    int nHalfKernelSize = m_nFilterKernelWidth/2;

    int nMaxDataSize = 800000;
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
            if(nImgCnt > 50){
                continue;
            }
            ++nImgCnt;

            if(mask.empty()){
                mask = cv::Mat_<uchar>(img.rows, img.cols, (uchar)255);
            }

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
    std::cout<<"start build sift dictionary\n";
    std::cout.flush();

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

            if(mask.empty()){
                mask = cv::Mat_<uchar>(img.rows, img.cols, (uchar)255);
            }

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
            VlDsiftFilter *dsift = vl_dsift_new_basic(nCol, nRow, 8, 12);

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

    //k-mean cluster to build sift dictionary
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

void MaterialClassifier::buildSiftIfvGMM(string dataAddress)
{
    std::cout<<"start build sift ifv dictionary\n";
    std::cout.flush();

    int nMaxDataSize = 3000000;
    int nSiftDim = 128;
    cv::Mat siftDescriptorSet = cv::Mat_<float>(nMaxDataSize, nSiftDim);

    //    Miscellaneous::IO::readData_<float>(siftDescriptorSet,dataAddress+"/sift_descriptors_set");
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

            if(mask.empty()){
                mask = cv::Mat_<uchar>(img.rows, img.cols, (uchar)255);
            }

            for(int j = 0; j < 3; ++j){
                int nRow = img.rows*m_siftScale[j];
                int nCol = img.cols*m_siftScale[j];

                cv::Mat scaledImg;
                cv::resize(img, scaledImg, cv::Size(nCol, nRow));

                float imgArray[nRow*nCol];
                //convert to vlfeat's image format
                for(int k = 0; k < nRow; ++k){
                    uchar *srcPtr = scaledImg.ptr<uchar>(k);
                    float *dstPtr = imgArray + k*nCol;
                    for(int l = 0; l < nCol; ++l){
                        *dstPtr = (float)(*srcPtr);
                        ++dstPtr;
                        ++srcPtr;
                    }
                }

                //detect dense sift
                VlDsiftFilter *dsift = vl_dsift_new_basic(nCol, nRow, 8, m_nSiftBinNum);

                vl_dsift_process(dsift, imgArray);
                const float *fpDescriptorPtr = vl_dsift_get_descriptors(dsift);
                int nDescriptorSize = vl_dsift_get_descriptor_size(dsift);
                int nKeyPointNum = vl_dsift_get_keypoint_num(dsift);
                const VlDsiftKeypoint *keyPoints =  vl_dsift_get_keypoints(dsift);

                for(int k = 0; k < nKeyPointNum; ++k){
                    int x = keyPoints[k].x/m_siftScale[j];
                    int y = keyPoints[k].y/m_siftScale[j];
                    if(mask.at<uchar>(y, x) < 100){
                        continue;
                    }
                    //normalize

                    //add to data set
                    float *dstPtr = siftDescriptorSet.ptr<float>(nDataCnt);
                    const float *srcPtr = fpDescriptorPtr+k*nDescriptorSize;
                    float l1Norm = 0.01;
                    for(int l = 0; l < nDescriptorSize; ++l){
                        *dstPtr = srcPtr[l];
                        if((*dstPtr) > 0){
                            l1Norm = l1Norm + (*dstPtr);
                        }else{
                            l1Norm = l1Norm - (*dstPtr);
                        }
                        ++dstPtr;
                    }

                    dstPtr = siftDescriptorSet.ptr<float>(nDataCnt);
                    for(int l = 0; l < nDescriptorSize; ++l){
                        *dstPtr = (*dstPtr)/l1Norm;
                        ++dstPtr;
                    }
                    ++nDataCnt;
                }
                vl_dsift_delete(dsift);
            }
            //            Miscellaneous::IO::data2Text_<float>(siftDescriptor, imgSiftSaveAddress);
        }
        closedir(pDir);
    }

    //build gmm distribution
    siftDescriptorSet.adjustROI(0,nDataCnt-nMaxDataSize,0,0);

    Miscellaneous::IO::data2Text_<float>(siftDescriptorSet, dataAddress+"/sift_descriptors_set");

    //    cv::Mat siftDescriptorSet;
    //    Miscellaneous::IO::readData_<float>(siftDescriptorSet, dataAddress+"/sift_descriptors_set");
    int descriptorNum = siftDescriptorSet.rows;
    int descriptorSize = siftDescriptorSet.cols;
    float data[1000000];

    int maxNum = 1000000/128;
    for(int i = 0; i < maxNum; ++i){
        //        float *srcPtr = siftDescriptorSet.ptr<float>(i);
        //        float *dstPtr = data + i*siftDescriptorSet.cols;
        //        for(int j = 0; j < siftDescriptorSet.cols; ++j){
        //            *dstPtr = *srcPtr;
        //            ++dstPtr;
        //            ++srcPtr;
        //        }
        for(int j = 0; j <descriptorSize; ++j){
            data[i*descriptorSize + j] = siftDescriptorSet.at<float>(i,j);
        }
    }

    int dimension = nSiftDim;
    int numData = maxNum;
    int numClusters = ms_nSiftGMMClusterNum;

    VlGMM *gmm = vl_gmm_new (VL_TYPE_FLOAT, nSiftDim, numClusters) ;
    vl_gmm_cluster(gmm, data, numData);

    const void *gmmMeans = vl_gmm_get_means(gmm);
    const void *gmmCovariance = vl_gmm_get_covariances(gmm);
    const void *prior = vl_gmm_get_priors(gmm);

    cv::Mat siftGmmInfo = cv::Mat_<float>(numClusters, dimension+dimension+1);//mean + digonal covariance + prior

    for(int i = 0; i < numClusters; ++i){
        float *dstGmmPtr = siftGmmInfo.ptr<float>(i);
        const float *fGmmMeans = static_cast<const float*>(gmmMeans);
        const float *fGmmCov = static_cast<const float*>(gmmCovariance);
        const float *fGmmPrior = static_cast<const float*>(prior);

        for(int j = 0; j < dimension; ++j){
            *dstGmmPtr = fGmmMeans[i*dimension + j];
            ++dstGmmPtr;
        }

        for(int j = 0; j < dimension; ++j){
            *dstGmmPtr = fGmmCov[i*dimension + j];
            ++dstGmmPtr;
        }

        *dstGmmPtr = fGmmPrior[i];
    }

    Miscellaneous::IO::data2Text_<float>(siftGmmInfo, dataAddress+"/sift_gmm_info");

    m_siftGMMDist.load(std::string(dataAddress+"/sift_gmm_info"), dimension);
}

void MaterialClassifier::buildColorGMMDist(string dataAddress)
{
    int nMaxDataSize = 500000;
    int nColorPatchWidth = 3;
    int nHalfColorPatchWidth = 1;
    int nColorPatchLength = 27;
    cv::Mat colorPatchDataSet = cv::Mat_<float>(nMaxDataSize, nColorPatchLength);
    int nDataCnt = 0;

    int nColorPatchSize = 9;
    int shiftX[9], shiftY[9];
    {
        int cnt = 0;
        for(int i = -1; i < 2; ++i){
            for(int j = -1; j < 2; ++j){
                shiftX[cnt] = i;
                shiftY[cnt] = j;
                ++cnt;
            }

        }
    }
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

            if(mask.empty()){
                mask = cv::Mat_<uchar>(img.rows, img.cols, (uchar)255);
            }

            //sample response value
            int nCol = img.cols;
            int nRow = img.rows;
            int nMaxX = nCol - nHalfColorPatchWidth;
            int nMinX = nHalfColorPatchWidth;
            int nMaxY = nRow - nHalfColorPatchWidth;
            int nMinY = nHalfColorPatchWidth;
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

                    float *dstPtr = colorPatchDataSet.ptr<float>(nDataCnt);

                    for(int j = 0; j < nColorPatchSize; ++j){
                        cv::Vec3b bgr = img.at<cv::Vec3b>(y+shiftY[j], x+shiftX[j]);
                        *dstPtr = bgr[0]; ++dstPtr;
                        *dstPtr = bgr[1]; ++dstPtr;
                        *dstPtr = bgr[2]; ++dstPtr;

                    }

                    ++nDataCnt;
                }
            }
        }
        closedir(pDir);
    }

    colorPatchDataSet.adjustROI(0, nDataCnt-nMaxDataSize, 0, 0);

    //k-mean cluster to build texton

    //    filterResponseSet.adjustROI(0,nDataCnt-nMaxDataSize,0,0);
    //    int K = ms_nGlobalTextonDictionarySize;
    //    cv::TermCriteria critera;
    //    critera.type = cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS;
    //    critera.maxCount = 500;
    //    critera.epsilon = 0.1;
    //    cv::Mat bestLabels;
    //    cv::Mat centers;
    //    cv::kmeans(filterResponseSet, K, bestLabels, critera, 20, cv::KMEANS_PP_CENTERS, centers);

    //    m_globalTextonDictionary = centers;

    //    Miscellaneous::IO::data2Text_<float>(centers, dataAddress+"/texton_dictionary");

    int descriptorNum = colorPatchDataSet.rows;
    int descriptorSize = colorPatchDataSet.cols;

    int maxArraySize = std::min(descriptorNum*descriptorSize, 1000000);
    float data[maxArraySize];

    int maxNum = maxArraySize/descriptorSize;
    for(int i = 0; i < maxNum; ++i){
        for(int j = 0; j <descriptorSize; ++j){
            data[i*descriptorSize + j] = colorPatchDataSet.at<float>(i,j);
        }
    }

    int dimension = descriptorSize;
    int numData = maxNum;
    int numClusters = ms_nColorGMMClusterNum;

    VlGMM *gmm = vl_gmm_new (VL_TYPE_FLOAT, dimension, numClusters) ;
    vl_gmm_cluster(gmm, data, numData);

    const void *gmmMeans = vl_gmm_get_means(gmm);
    const void *gmmCovariance = vl_gmm_get_covariances(gmm);
    const void *prior = vl_gmm_get_priors(gmm);

    cv::Mat colorGmmInfo = cv::Mat_<float>(numClusters, dimension+dimension+1);//mean + digonal covariance + prior

    for(int i = 0; i < numClusters; ++i){
        float *dstGmmPtr = colorGmmInfo.ptr<float>(i);
        const float *fGmmMeans = static_cast<const float*>(gmmMeans);
        const float *fGmmCov = static_cast<const float*>(gmmCovariance);
        const float *fGmmPrior = static_cast<const float*>(prior);

        for(int j = 0; j < dimension; ++j){
            *dstGmmPtr = fGmmMeans[i*dimension + j];
            ++dstGmmPtr;
        }

        for(int j = 0; j < dimension; ++j){
            *dstGmmPtr = fGmmCov[i*dimension + j];
            ++dstGmmPtr;
        }

        *dstGmmPtr = fGmmPrior[i];
    }

    Miscellaneous::IO::data2Text_<float>(colorGmmInfo, dataAddress+"/color_gmm_info");

    m_colorGMMDist.load(std::string(dataAddress+"/color_gmm_info"), dimension);

}

void MaterialClassifier::buildColorModelSet(string dataAddress)
{

}

void MaterialClassifier::buildModelSet(string dataAddress)
{

}

size_t MaterialClassifier::ms_nGlobalTextonDictionarySize = 256;//384
size_t MaterialClassifier::ms_nColorDictionarySize = 64;
size_t MaterialClassifier::ms_nSiftDictionarySize = 128;
size_t MaterialClassifier::ms_nSiftGMMClusterNum = 128;
size_t MaterialClassifier::ms_nColorGMMClusterNum = 128;
size_t MaterialClassifier::ms_nTextonGMMClusterNum = 96;
size_t MaterialClassifier::ms_nSiftIFVDimension = 60;
size_t MaterialClassifier::ms_nColorIFVDimension = 100;
size_t MaterialClassifier::ms_nTextonIFVDimension = 100;
