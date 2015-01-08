#include "multimaterial_classifier.h"

MultiMaterialClassifier::MultiMaterialClassifier()
    : m_nMaterialTypes(5),
      m_nFilterKernelSize(7),
      m_nFilterKernelSetSize(60)
{
    m_tableMaterialColor.insert(std::pair<int, cv::Vec3b>(0, cv::Vec3b(255, 0, 0)));
    m_tableMaterialColor.insert(std::pair<int, cv::Vec3b>(1, cv::Vec3b(0, 255, 0)));
    m_tableMaterialColor.insert(std::pair<int, cv::Vec3b>(2, cv::Vec3b(0, 0, 255)));
    m_tableMaterialColor.insert(std::pair<int, cv::Vec3b>(3, cv::Vec3b(255, 255, 0)));
    m_tableMaterialColor.insert(std::pair<int, cv::Vec3b>(4, cv::Vec3b(0, 255, 255)));
    //    m_tableMaterialColor.insert(std::pair<int, cv::Vec3b>(5, cv::Vec3b(255, 0, 255)));
    //    m_tableMaterialColor.insert(std::pair<int, cv::Vec3b>(6, cv::Vec3b(64, 128, 0)));

    //0 others
    //1 green plant
    //2 wall
    //3 road
    //4 sky
    m_tableMaterialName.insert(std::pair<int, cv::String>(0, "others"));
    m_tableMaterialName.insert(std::pair<int, cv::String>(1, "green plant"));
    m_tableMaterialName.insert(std::pair<int, cv::String>(2, "wall"));
    m_tableMaterialName.insert(std::pair<int, cv::String>(3, "road"));
    m_tableMaterialName.insert(std::pair<int, cv::String>(4, "sky"));

    m_vecClassModelSize.resize(m_nMaterialTypes);
    m_vecClassModelSize[0] = 5;
    m_vecClassModelSize[1] = 10;
    m_vecClassModelSize[2] = 7;
    m_vecClassModelSize[3] = 4;
    m_vecClassModelSize[4] = 2;

    bUseTexton = true;
    bUseChroma = true;
    bUseSpectrum = true;
    bUseNdvi = true;

}

void MultiMaterialClassifier::smoothMaterialMap(Mat &inputMap, Mat &outputMap)
{
    outputMap = inputMap;
}

void MultiMaterialClassifier::visualizeMaterialMap(Mat &materialMap, Mat &_visualImg)
{
    int Rows = materialMap.rows;
    int Cols = materialMap.cols;

    cv::Mat visualImg = cv::Mat_<cv::Vec3b>(Rows, Cols);

    uchar *materialPtr;
    for(int i = 0; i < Rows; ++i){
        materialPtr = materialMap.ptr<uchar>(i);
        for(int j = 0; j < Cols; ++j){
            cv::Vec3b color= m_tableMaterialColor[(int)(*materialPtr++)];
            visualImg.at<cv::Vec3b>(i,j) = color;
        }
    }

    _visualImg = visualImg;
}

void MultiMaterialClassifier::getColorBar(Mat &colorBar)
{
    int colorTypes = m_tableMaterialColor.size();
    int singleBarHeight = 30;
    int singleBarWidth = 100;
    cv::Mat bar = cv::Mat_<cv::Vec3b>(colorTypes*singleBarHeight, singleBarWidth);

    for(int i = 0; i < colorTypes; ++i){
        cv::Vec3b *colorBarPtr;
        for(int j = 0; j < singleBarHeight; ++j){
            colorBarPtr = bar.ptr<cv::Vec3b>(i*singleBarHeight+j);
            cv::Vec3b color = m_tableMaterialColor[i];
            for(int k = 0; k < singleBarWidth; ++k){
                *colorBarPtr++ = color;
            }
        }
    }

    for(int i = 0; i < colorTypes; ++i){
        cv::String text = m_tableMaterialName[i];
        cv::putText(bar, text, cv::Point(0,(i+1)*singleBarHeight-5), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,0,0));
    }

    colorBar = bar;
}

void MultiMaterialClassifier::train(QString dataAddress)
{
    bool trainKernel = false;
    bool trainDictionary = false;
    bool trainModel = true;

    if(trainKernel){
        buildFilterKernelSet(dataAddress);
        //        buildGaborFilterKernelSet();
    }else{
        cv::Mat centers;
        Miscellaneous::IO::readData_<float>(centers, "kernel_set",false);
        m_nFilterKernelSetSize = centers.rows;
        m_nFilterKernelSize = static_cast<size_t>(std::sqrt(centers.cols));
        if(m_vecFilterKernelSet.size() != 0){
            m_vecFilterKernelSet.clear();
        }
        m_vecFilterKernelSet.resize(m_nFilterKernelSetSize);

        for(size_t i = 0; i < centers.rows; ++i){
            m_vecFilterKernelSet[i] = cv::Mat_<float>(m_nFilterKernelSize, m_nFilterKernelSize);
            for(int j = 0; j < m_nFilterKernelSize; ++j){
                for(int k = 0; k < m_nFilterKernelSize; ++k){
                    (m_vecFilterKernelSet[i]).at<float>(j,k) = centers.at<float>(i, j*m_nFilterKernelSize+k);
                }
            }
        }
    }

    if(trainDictionary){
        buildGlobalTextonDictionary(dataAddress);
    }else{
        cv::Mat dictionary;
        QString fileName = "global_texton_dictionary";
        Miscellaneous::IO::readData_<float>(dictionary, fileName, false);
        m_globalTextonDictionary = dictionary;
        ms_nGlobalTextonDictionarySize = dictionary.rows;
    }

    if(trainModel){
        buildModelSet(dataAddress);
        //          UnsupervisedModelBuilding(dataAddress);
    }else{
        int categoryNum = m_nMaterialTypes;
        m_vecClassVecModelSet.clear();
        m_vecClassVecModelSet.resize(categoryNum);
        m_vecClassModelSize.clear();
        m_vecClassModelSize.resize(categoryNum);

        for(int i = 0; i < categoryNum; ++i){
            cv::Mat classModel;
            QString classModelFileName = "model_"+QString::number(i);
            Miscellaneous::IO::readData_<float>(classModel, classModelFileName, false);
            m_vecClassVecModelSet[i] = classModel;
            m_vecClassModelSize[i] = classModel.rows;
        }
    }
}

void MultiMaterialClassifier::load()
{
    //load filter kernel
    cv::Mat centers;
    Miscellaneous::IO::readData_<float>(centers, "kernel_set",false);
    m_nFilterKernelSetSize = centers.rows;
    m_nFilterKernelSize = static_cast<size_t>(std::sqrt(centers.cols));
    if(m_vecFilterKernelSet.size() != 0){
        m_vecFilterKernelSet.clear();
    }
    m_vecFilterKernelSet.resize(m_nFilterKernelSetSize);

    for(size_t i = 0; i < centers.rows; ++i){
        m_vecFilterKernelSet[i] = cv::Mat_<float>(m_nFilterKernelSize, m_nFilterKernelSize);
        for(int j = 0; j < m_nFilterKernelSize; ++j){
            for(int k = 0; k < m_nFilterKernelSize; ++k){
                (m_vecFilterKernelSet[i]).at<float>(j,k) = centers.at<float>(i, j*m_nFilterKernelSize+k);
            }
        }
    }


    //load texton
    cv::Mat dictionary;
    QString fileName = "global_texton_dictionary";
    Miscellaneous::IO::readData_<float>(dictionary, fileName, false);
    m_globalTextonDictionary = dictionary;
    ms_nGlobalTextonDictionarySize = dictionary.rows;


    //load model
    int categoryNum = m_nMaterialTypes;
    //    m_vecClassVecModelSet.clear();
    m_vecClassVecModelSet.resize(categoryNum);
    //    m_vecClassModelSize.clear();
    m_vecClassModelSize.resize(categoryNum);
    //    m_vecClassMaterialModelSets.clear();
    m_vecClassMaterialModelSets.resize(categoryNum);

    for(int i = 0; i < categoryNum; ++i){
        cv::Mat classVecModel,transformSet;
        QString classModelFileName = "model_"+QString::number(i);
        QString classModelTransformFileName = "T_"+QString::number(i);
        Miscellaneous::IO::readData_<float>(classVecModel, classModelFileName, false);
        Miscellaneous::IO::readData_<float>(transformSet, classModelTransformFileName, false);
        m_vecClassVecModelSet[i] = classVecModel;
        m_vecClassModelSize[i] = classVecModel.rows;

        //convert to MaterialFeatureSet format
        std::vector<MaterialFeatureSet> materialModelSet;
        for(int j = 0; j < classVecModel.rows; ++j){
            float *srcPtr = classVecModel.ptr<float>(j);
            MaterialFeatureSet feature;
            feature.vecMaterialFeatures.resize(4);

            //textonDictionary
            feature.textonHistogram = cv::Mat_<float>(1, ms_nGlobalTextonDictionarySize);
            float *dstPtr = feature.textonHistogram.ptr<float>(0);
            for(int k = 0; k < ms_nGlobalTextonDictionarySize; ++k){
                *dstPtr++ = *srcPtr++;
            }
            feature.vecMaterialFeatures[0] = feature.textonHistogram;

            //spectrum
            feature.spectrumResponse = cv::Mat_<float>(1, ms_nSpectrumNum);
            dstPtr = feature.spectrumResponse.ptr<float>(0);
            for(int k = 0; k < ms_nSpectrumNum; ++k){
                *dstPtr++ = *srcPtr++;
            }
            feature.vecMaterialFeatures[1] = feature.spectrumResponse;

            //ndvi
            feature.ndviHistogram = cv::Mat_<float>(1, ms_nNdviHistNum);
            dstPtr = feature.ndviHistogram.ptr<float>(0);
            for(int k = 0; k < ms_nNdviHistNum; ++k){
                *dstPtr++ = *srcPtr++;
            }
            feature.vecMaterialFeatures[2] = feature.ndviHistogram;

            //chroma
            feature.chromaHistogram = cv::Mat_<float>(1, ms_nChromaHistNum);
            dstPtr = feature.chromaHistogram.ptr<float>(0);
            for(int k = 0; k < ms_nChromaHistNum; ++k){
                *dstPtr++ = *srcPtr++;
            }
            feature.vecMaterialFeatures[3] = feature.chromaHistogram;

            //compute transform matrix
            cv::Mat T = cv::Mat_<float>(3,1);
            for(int k = 0; k < 3; k++){
                T.at<float>(k,0) =  transformSet.at<float>(j,k);
            }
            std::cout<<"Transform matrix T\n"<<T<<std::endl;
            feature.rgb2irMat = T;

            //push model to buffer
            materialModelSet.push_back(feature);
        }
        m_vecClassMaterialModelSets[i] = materialModelSet;
    }

}

void MultiMaterialClassifier::test(StereoData &data)
{
    int categoryNum = m_nMaterialTypes;
    int nCol = data.baseImg.cols;
    int nRow = data.baseImg.rows;

    SLICSuperpixel slic;
    slic.init(data.baseImg, 150, 50, 40);
    slic.generateSuperPixels();
    cv::Mat contourImg = data.baseImg.clone();
    /* Draw the contours bordering the clusters */
    vector<Point2i> contours = slic.getContours();
    for( Point2i contour: contours )
        contourImg.at<Vec3b>( contour.y, contour.x ) = Vec3b(255, 0, 255);

    std::vector<MaterialFeatureSet> clusterFeatureSet;
    extractClusterFeature(data, slic, clusterFeatureSet);

    //for each superpixel, determine its material
    MaterialFeatureDist<float> dist;
    dist.setFeatureUtility(bUseTexton, bUseSpectrum, bUseNdvi, bUseChroma);

    std::vector<cv::Point2i> clusterCenters = slic.getClusterCenters();
    int clusterNum = clusterCenters.size();
    std::map<int, int> cluster2materialTable;
    std::map<int, int> cluster2modelIndexTable;
    for(int i = 0; i < clusterNum; ++i){
        QString stringIndex = QString::number(i);
        cv::putText(contourImg,stringIndex.toStdString(), clusterCenters[i],FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,0,0));
        std::cout<<"match patch "<<i<<std::endl;
        //voting for each pixel
        cv::Mat dataVector;
        concatenateVector2Mat(clusterFeatureSet[i].vecMaterialFeatures, dataVector);
        float *dataPtr = dataVector.ptr<float>(0);

        //match with model
        int classifyResult = 0;
        int minDistClass = 0;
        int minDistModelIndex = 0;
        double minDist = std::numeric_limits<float>::max();
        for(int j = 0; j < categoryNum; ++j){
            std::cout<<"matching material "<<j<<" ";
            for(int k = 0; k < m_vecClassVecModelSet[j].rows; ++k){
                float *modelPtr = m_vecClassVecModelSet[j].ptr<float>(k);
                double totalDist = dist(modelPtr, dataPtr, 20);
                std::cout<<totalDist<<' ';

                if(totalDist < minDist){
                    minDist = totalDist;
                    minDistClass = j;
                    minDistModelIndex = k;
                }
            }
            std::cout<<"\n";
        }
        std::cout<<"final result: class "<<minDistClass<<" model "<<minDistModelIndex<<std::endl;
        std::cout<<"\n";

        classifyResult =  minDistClass;

        cluster2materialTable.insert(std::pair<int, int>(i, classifyResult));
        cluster2modelIndexTable.insert(std::pair<int, int>(i, minDistModelIndex));
    }

    cv::imshow("contourImg", contourImg);

    cv::Mat materialMap = slic.getClustersIndex();
    cv::Mat convertedImg = cv::Mat_<float>(nRow, nCol);
    uchar *orgImgPtr;
    int *materialPtr;
    float *cvtImgPtr;
    for(int i = 0; i < nRow; ++i){
        orgImgPtr = data.baseImg.ptr<uchar>(i);
        materialPtr = materialMap.ptr<int>(i);
        cvtImgPtr = convertedImg.ptr<float>(i);
        for(int j = 0; j < nCol; ++j){
            int materialIndex = cluster2materialTable[*materialPtr];
            int modelIndex = cluster2modelIndexTable[*materialPtr];
            *materialPtr = materialIndex;
            cv::Mat bgr = cv::Mat_<float>(1,3);
            bgr.at<float>(0,0) = orgImgPtr[0];
            bgr.at<float>(0,1) = orgImgPtr[1];
            bgr.at<float>(0,2) = orgImgPtr[2];

            cv::Mat ir = bgr*m_vecClassMaterialModelSets[materialIndex][modelIndex].rgb2irMat;
            *cvtImgPtr = ir.at<float>(0,0);

            orgImgPtr += 3;
            cvtImgPtr++;
            materialPtr++;
        }
    }

    double max,min;
    cv::minMaxLoc(convertedImg, &min, &max);
    cv::Mat rescaledCIR = (convertedImg-min)/(max - min);
    cv::imshow("rescaled CIR", rescaledCIR);
    cv::Mat saveImg;
    convertedImg.convertTo(saveImg, CV_8UC3);
    cv::imwrite("right_cir.png", saveImg);
    cv::imwrite("left.png", data.referImg);
    cv::Mat gray;
    cv::cvtColor(data.baseImg, gray, cv::COLOR_BGR2GRAY);
    cv::imshow("gray", gray);
    cv::imwrite("right_gray.png", gray);
    cv::imshow("convertedIR", convertedImg/256);
    cv::imshow("org ir", data.referImg);
    materialMap.convertTo(m_finalMaterialMap, CV_8U);
    visualizeMaterialMap(m_finalMaterialMap, m_visualMaterialMap);
}

void MultiMaterialClassifier::buildFilterKernelSet(QString dataAddress)
{
    QStringList imgDirList = Miscellaneous::loadImageList(dataAddress);
    if(imgDirList.isEmpty()){
        return;
    }

    int kernelLength = m_nFilterKernelSize * m_nFilterKernelSize;
    int halfKernelSize = m_nFilterKernelSize/2;

    int shiftX[kernelLength];
    int shiftY[kernelLength];

    for(int i = 0; i < m_nFilterKernelSize; ++i){
        for(int j = 0; j < m_nFilterKernelSize; ++j){
            shiftX[i*m_nFilterKernelSize+j] = i- halfKernelSize;
            shiftY[i*m_nFilterKernelSize+j] = j - halfKernelSize;
        }
    }

    cv::Mat patchSet;

    int maxDataSize = 100000;
    int dataCnt = 0;

    XtionFull xiton;
    Rectification rectifier(xiton);
    for(int i = 0; i < imgDirList.size();){
        StereoData data;
        cv::Mat baseImgOrg, referImgOrg, baseDepthImgOrg, baseMaterialImgOrg, baseImgOrgDisp;
        baseDepthImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data()); i++;
        referImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data()); i++;
        baseImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data());i++;
        baseMaterialImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data()); i++;

        cv::Rect validRoiBase, validRoiRefer;
        rectifier.rectify(baseImgOrg, data.baseImg, validRoiBase, RECT_IMAGE_TYPE_RIGHT, cv::INTER_LANCZOS4);
        rectifier.rectify(referImgOrg, data.referImg, validRoiRefer, RECT_IMAGE_TYPE_LEFT, cv::INTER_LANCZOS4);
        rectifier.rectify(baseDepthImgOrg, data.baseDepth, validRoiBase, RECT_IMAGE_TYPE_RIGHT, cv::INTER_NEAREST);
        rectifier.rectify(baseMaterialImgOrg, data.baseMaterial, validRoiBase, RECT_IMAGE_TYPE_RIGHT, cv::INTER_NEAREST);
        Miscellaneous::Conversion::depth2horDisp_<ushort, float>(data.baseDepth, data.baseOrgDisp);

        if(patchSet.empty()){
            patchSet = cv::Mat_<float>(maxDataSize, kernelLength);
        }

        cv::Mat grayImg;
        cv::cvtColor(data.baseImg, grayImg, cv::COLOR_BGR2GRAY);

        int n_row = data.baseImg.rows;
        int n_col = data.baseImg.cols;

        float *dispPtr;
        for(int y = 4; y < (n_row-4); y+=1){
            dispPtr = data.baseOrgDisp.ptr<float>(y);
            for(int x = 4; x < (n_col-4); x+=1){
                float lucky = std::rand()*1.0/RAND_MAX;
                if(lucky > 0.015){
                    continue;
                }

                float *dstPtr = patchSet.ptr<float>(dataCnt);

                for(int k = 0; k < kernelLength; ++k){
                    *dstPtr = grayImg.at<uchar>(y+shiftY[k],x+shiftX[k]);
                    dstPtr++;
                }

                dataCnt++;
            }
        }
    }

    patchSet.adjustROI(0,dataCnt-maxDataSize,0,0);
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
        cv::Mat kernelImg = cv::Mat_<uchar>(m_nFilterKernelSize,m_nFilterKernelSize);
        double min, max;
        cv::minMaxLoc(centers.row(i), &min, &max);
        float scale = 254/(max-min);
        for(int j = 0; j < m_nFilterKernelSize; ++j){
            for(int k = 0; k < m_nFilterKernelSize; ++k){
                kernelImg.at<uchar>(j,k) = static_cast<uchar>(scale*centers.at<float>(i, j*m_nFilterKernelSize+k));
            }
        }
        QString name = QString::number(i);
        cv::imshow(name.toLocal8Bit().data(), kernelImg);
    }

    //zero mean and scale
    for(int i = 0; i < centers.rows; ++i){
        cv::Mat rowVec = centers.row(i);
        double min, max, scale, shift;
        cv::minMaxLoc(rowVec, &min, &max);
        scale = 1.0/(max-min);

        cv::Scalar meanValue = cv::mean(rowVec);
        shift = meanValue[0];

        rowVec  = rowVec-shift;
        rowVec = rowVec*scale;
    }

    if(m_vecFilterKernelSet.size() != 0){
        m_vecFilterKernelSet.clear();
    }
    m_vecFilterKernelSet.resize(m_nFilterKernelSetSize);

    for(size_t i = 0; i < centers.rows; ++i){
        m_vecFilterKernelSet[i] = cv::Mat_<float>(m_nFilterKernelSize, m_nFilterKernelSize);
        for(int j = 0; j < m_nFilterKernelSize; ++j){
            for(int k = 0; k < m_nFilterKernelSize; ++k){
                (m_vecFilterKernelSet[i]).at<float>(j,k) = centers.at<float>(i, j*m_nFilterKernelSize+k);
            }
        }
    }
    Miscellaneous::IO::data2Text_<float>(centers, "kernel_set");

    cv::waitKey(5);
}

void MultiMaterialClassifier::buildGaborFilterKernelSet()
{
    int gaborSize = 64;

    float theta[8] = {0, M_PI/8, M_PI/4,3*M_PI/8, M_PI/2, 5*M_PI/8, 6*M_PI/8, 7*M_PI/8};
    float sigma[2] = {M_PI/2,3*M_PI/4};
    float lambd[2] = {M_PI/4, M_PI};
    float gamma[2] = {2, 3};
    cv::Mat centers = cv::Mat_<float>(m_nFilterKernelSetSize, m_nFilterKernelSize*m_nFilterKernelSize);

    std::vector<cv::Mat> kernels(64);
    for(int i = 0; i < 8; ++i){ //theta
        for(int j = 0; j < 2; ++j){ //lambd
            for(int k = 0; k < 2; ++k){ //gamma
                for(int l = 0; l < 2; ++l){ //sigma
                    kernels[i*8+j*4+k*2+l]=cv::getGaborKernel(cv::Size(m_nFilterKernelSize,m_nFilterKernelSize),sigma[l],
                                                              theta[i], lambd[j],
                                                              gamma[k],0, CV_32F);
                    //                    std::cout<<"sigma:"<<sigma[l]<<" theta:"<<theta[i]<<" lambd:"<<lambd[j]<<" gamma:"<<gamma[k]<<std::endl;

                }
            }
        }
    }

    if(m_vecFilterKernelSet.size() != 0){
        m_vecFilterKernelSet.clear();
    }
    m_vecFilterKernelSet.resize(m_nFilterKernelSetSize);

    for(size_t i = 0; i < centers.rows; ++i){
        m_vecFilterKernelSet[i] = kernels[i];

        for(int j = 0; j < m_nFilterKernelSize; ++j){
            for(int k = 0; k < m_nFilterKernelSize; ++k){
                centers.at<float>(i, j*m_nFilterKernelSize+k) = kernels[i].at<float>(j,k);
            }
        }
    }
    Miscellaneous::IO::data2Text_<float>(centers, "kernel_set");
}

void MultiMaterialClassifier::buildGlobalTextonDictionary(QString dataAddress)
{
    QStringList imgDirList = Miscellaneous::loadImageList(dataAddress);
    if(imgDirList.isEmpty()){
        return;
    }

    size_t nGlobalTextonDictionarySize = ms_nGlobalTextonDictionarySize;
    int maxDataSize = 100000;
    cv::Mat materialTrainData = cv::Mat_<float>(maxDataSize, m_nFilterKernelSetSize, 0.0f);
    int totalCnt = 0;

    XtionFull xiton;
    Rectification rectifier(xiton);

    for(int i = 0; i < imgDirList.size();){
        StereoData data;
        cv::Mat baseImgOrg, referImgOrg, baseDepthImgOrg, baseMaterialImgOrg;
        baseDepthImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data()); i++;
        referImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data()); i++;
        baseImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data());i++;
        baseMaterialImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data()); i++;

        cv::Rect validRoiBase, validRoiRefer;
        rectifier.rectify(baseImgOrg, data.baseImg, validRoiBase, RECT_IMAGE_TYPE_RIGHT, cv::INTER_LANCZOS4);
        rectifier.rectify(referImgOrg, data.referImg, validRoiRefer, RECT_IMAGE_TYPE_LEFT, cv::INTER_LANCZOS4);
        rectifier.rectify(baseDepthImgOrg, data.baseDepth, validRoiBase, RECT_IMAGE_TYPE_RIGHT, cv::INTER_NEAREST);
        rectifier.rectify(baseMaterialImgOrg, data.baseMaterial, validRoiBase, RECT_IMAGE_TYPE_RIGHT, cv::INTER_NEAREST);
        Miscellaneous::Conversion::depth2horDisp_<ushort, float>(data.baseDepth, data.baseOrgDisp);

        cv::Mat totalResponse;
        //compute filtering response
        {
            cv::Mat grayImg, f_grayImg;
            cv::cvtColor(data.baseImg, grayImg, cv::COLOR_BGR2GRAY);
            grayImg.convertTo(f_grayImg, CV_32F);

            std::vector<cv::Mat> responseMap(m_nFilterKernelSetSize);
            for(int j = 0; j < m_nFilterKernelSetSize; ++j){
                cv::filter2D(f_grayImg, responseMap[j], CV_32F, m_vecFilterKernelSet[j]);
            }

            cv::merge(responseMap, totalResponse);
        }

        int n_row =  data.baseImg.rows;
        int n_col =  data.baseImg.cols;

        float *dispPtr;
        for(int y = 4; y < (n_row-4); y+=1){
            dispPtr =  data.baseOrgDisp.ptr<float>(y);
            for(int x = 4; x < (n_col-4); x+=1){
                float lucky = std::rand()*1.0/RAND_MAX;
                if(lucky > 0.015){
                    continue;
                }

                float *srcPtr = totalResponse.ptr<float>(y) + x*m_nFilterKernelSetSize;
                float *dstPtr = materialTrainData.ptr<float>(totalCnt);
                for(int k = 0; k < m_nFilterKernelSetSize; ++k){
                    *dstPtr = *srcPtr;
                    dstPtr++;
                    srcPtr++;
                }

                totalCnt += 1;
            }
        }
    }

    cv::Mat globalTextonDictionary;
    {
        materialTrainData.adjustROI(0, totalCnt-maxDataSize, 0, 0);
        std::cout<<"train global dictionary  and train data size: "<<totalCnt<<std::endl;
        int desiredK = nGlobalTextonDictionarySize;
        cvflann::KMeansIndexParams kmeanParam(5, 100, cvflann::FLANN_CENTERS_RANDOM);
        globalTextonDictionary = cv::Mat_<float>(desiredK, m_nFilterKernelSetSize);
        int foundCluster = cv::flann::hierarchicalClustering<cvflann::L2<float> >(materialTrainData,
                                                                                  globalTextonDictionary,
                                                                                  kmeanParam);
        globalTextonDictionary.adjustROI(0, foundCluster - desiredK, 0, 0);

        //        cv::TermCriteria critera;
        //        critera.type = cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS;
        //        critera.maxCount = 500;
        //        critera.epsilon = 0.1;
        //        cv::Mat bestLabels;
        //        cv::Mat centers;
        //        cv::BOWKMeansTrainer bow(desiredK, critera, 20, cv::KMEANS_RANDOM_CENTERS);
        //        bow.add(materialTrainData);
        //        centers = bow.cluster();
        //        globalTextonDictionary = centers;
        //        int foundCluster = centers.rows;
        ms_nGlobalTextonDictionarySize = foundCluster;
    }

    m_globalTextonDictionary = globalTextonDictionary;
    QString savefileName = "global_texton_dictionary";
    Miscellaneous::IO::data2Text_<float>(globalTextonDictionary, savefileName.toStdString());
}

void MultiMaterialClassifier::buildModelSet(QString dataAddress)
{
    std::cout<<"start build model set\n";
    std::cout.flush();
    QStringList imgDirList = Miscellaneous::loadImageList(dataAddress);
    if(imgDirList.isEmpty()){
        return;
    }

    int clusterNum = 200;
    std::vector<size_t> classModelSize = m_vecClassModelSize;
    int maxSingleClassSize = 700;
    int categoryNum = m_nMaterialTypes;
    std::vector<cv::Mat> materialTrainData(categoryNum);

    int classCnt[categoryNum];
    for(int i = 0; i < categoryNum; ++i){
        classCnt[i] = 0;
    }

    SLICSuperpixel slic;
    XtionFull device;
    Rectification rectifier(device);
    for(int i = 0; i < imgDirList.size();){
        cv::Mat baseImgOrg, referImgOrg, baseDepthImgOrg, baseMaterialImgOrg;
        StereoData data;
        baseDepthImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data()); i++;
        referImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data(),cv::IMREAD_GRAYSCALE); i++;
        baseImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data());i++;
        baseMaterialImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data(), cv::IMREAD_GRAYSCALE); i++;

        cv::Rect validRoiBase, validRoiRefer;
        rectifier.rectify(baseImgOrg, data.baseImg, validRoiBase, RECT_IMAGE_TYPE_RIGHT, cv::INTER_LANCZOS4);
        rectifier.rectify(referImgOrg, data.referImg, validRoiRefer, RECT_IMAGE_TYPE_LEFT, cv::INTER_LANCZOS4);
        rectifier.rectify(baseDepthImgOrg, data.baseDepth, validRoiBase, RECT_IMAGE_TYPE_RIGHT, cv::INTER_NEAREST);
        rectifier.rectify(baseMaterialImgOrg, data.baseMaterial, validRoiBase, RECT_IMAGE_TYPE_RIGHT, cv::INTER_NEAREST);
        Miscellaneous::Conversion::depth2horDisp_<ushort, float>(data.baseDepth, data.baseOrgDisp);

        slic.init(data.baseImg, 150, 80, 40);
        slic.generateSuperPixels();

        //        cv::Mat contourImg = data.baseImg.clone();
        //        /* Draw the contours bordering the clusters */
        //        vector<Point2i> contours = slic.getContours();
        //        for( Point2i contour: contours ){
        //            contourImg.at<Vec3b>( contour.y, contour.x ) = Vec3b(255, 0, 255);
        //        }
        //        std::cout<<data.baseMaterial.channels()<<std::endl;
        //        cv::imshow("contours", contourImg);
        //        cv::imshow("material", data.baseMaterial*60);
        //        cv::waitKey(0);

        std::vector<MaterialFeatureSet> clusterFeatureSet;
        extractClusterFeature(data, slic, clusterFeatureSet);

        if(materialTrainData[0].empty()){
            int nTotalLength = 0;
            for(int j = 0; j < clusterFeatureSet[0].vecMaterialFeatures.size(); ++j){
                nTotalLength += clusterFeatureSet[0].vecMaterialFeatures[j].cols;
            }
            for(int j = 0; j < categoryNum; ++j){
                materialTrainData[j] = cv::Mat_<float>(maxSingleClassSize, nTotalLength, 0.f);
            }
        }

        std::vector<cv::Point2i> clusterCenters = slic.getClusterCenters();
        cv::Mat clusterIndexMap = slic.getClustersIndex();
        clusterNum = clusterCenters.size();

        for(int j = 0; j < clusterNum; ++j){
            uchar material;
            cv::Mat mask;
            cv::Rect patchRoi;
            getClusterMask(clusterIndexMap, clusterCenters[j], j, mask, patchRoi);
            int baseX = patchRoi.x;
            int baseY = patchRoi.y;

            //            cv::imshow("patch", mask);
            //            cv::waitKey();

            //vote for material type
            std::vector<int> voteBox(m_nMaterialTypes, 0);
            int clusterCnt = 0;
            for(int k = 0; k < mask.rows; ++k){
                int y = k+baseY;
                for(int l = 0; l < mask.cols; ++l){
                    uchar ucMaskValue = mask.at<uchar>(k, l);
                    if(ucMaskValue < 200){
                        continue;
                    }
                    int x = l+baseX;
                    int materialType = data.baseMaterial.at<uchar>(y, x);

                    voteBox[materialType] += 1;
                    clusterCnt++;
                }
            }

            //select the material type
            {
                int maxCnt = 0;
                int maxMaterialBin = 0;
                for(int k = 0; k < voteBox.size(); ++k){
                    if(voteBox[k] > maxCnt){
                        maxCnt = voteBox[k];
                        maxMaterialBin = k;
                    }
                }
                material = maxMaterialBin;
            }

            if(classCnt[material] >= maxSingleClassSize){
                continue;
            }

            float *dstPtr = materialTrainData[material].ptr<float>(classCnt[material]);
            for(int k = 0; k < clusterFeatureSet[j].vecMaterialFeatures.size(); ++k){
                float *srcPtr = clusterFeatureSet[j].vecMaterialFeatures[k].ptr<float>(0);
                int featureLength = clusterFeatureSet[j].vecMaterialFeatures[k].cols;
                for(int l = 0; l < featureLength; ++l){
                    *dstPtr++ = *srcPtr++;
                }
            }
            //            std::cout<<j<<" "<<i<<" "<<clusterFeatureSet[j].vecMaterialFeatures[2]<<std::endl;
            classCnt[material] += 1;
        }
        for(int i = 0; i < categoryNum; ++i){
            std::cout<<classCnt[i]<<" ";
        }
        std::cout<<std::endl;
        slic.clear();
    }

    std::vector<cv::Mat> classModelSet(categoryNum);
    m_vecClassVecModelSet.resize(categoryNum);
    m_vecClassMaterialModelSets.resize(categoryNum);
    m_vecRgb2IrTransformSet.resize(categoryNum);
    for(int i = 0; i < categoryNum; ++i){
        size_t desiredClusterNum  = classModelSize[i];
        materialTrainData[i].adjustROI(0, classCnt[i]-maxSingleClassSize, 0, 0);


        cvflann::KMeansIndexParams kmeanParam(3, 100, cvflann::CENTERS_RANDOM);
        classModelSet[i] = cv::Mat_<float>(desiredClusterNum, materialTrainData[i].cols);

        MaterialFeatureDist<float> dist;
        dist.setFeatureUtility(bUseTexton, bUseSpectrum, bUseNdvi, bUseChroma);
        int foundCluster = cv::flann::hierarchicalClustering<MaterialFeatureDist<float> >(materialTrainData[i],
                                                                                          classModelSet[i],
                                                                                          kmeanParam,
                                                                                          dist);
        std::cout<<"material "<<i<<" contain "<<materialTrainData[i].rows<<" train data\n";
        std::cout.flush();
        classModelSet[i].adjustROI(0, foundCluster-desiredClusterNum, 0, 0);

        //for each submodel found its label
        cv::Mat bestLabels = cv::Mat_<int>(materialTrainData[i].rows, 1);
        for(int j = 0; j < materialTrainData[i].rows; ++j){
            float *dataPtr = materialTrainData[i].ptr<float>(j);
            int nMinDistTextonIndex = 0;
            double dMinDist = std::numeric_limits<float>::max();
            for(int k = 0; k < classModelSet[i].rows; ++k){
                float *modelPtr = classModelSet[i].ptr<float>(k);
                double dDist = dist(dataPtr, modelPtr, 20);
                if(dDist < dMinDist){
                    nMinDistTextonIndex = k;
                    dMinDist = dDist;
                }
            }
            bestLabels.at<int>(j,0) = nMinDistTextonIndex;
        }
        //        cv::flann::Index flannIndex(classModelSet[i],cv::flann::LinearIndexParams(),cvflann::FLANN_DIST_EUCLIDEAN);
        //        flannIndex.knnSearch(materialTrainData[i], bestLabels, distMat, 1);

        // using mlpack's kmean method
        //        arma::fmat armaData;
        //        Miscellaneous::Conversion::cvmat2armamat_<float, float>(materialTrainData[i], armaData);
        //        armaData = armaData.t();
        //        arma::fmat centroids;
        //        arma::Col<size_t> assignments;
        //        typedef MaterialFeatureMetric MaterialMetric;
        //        mlpack::kmeans::KMeans<MaterialMetric> kmean;

        // use traditonal kmean in opencv
        //        //        hierarchyKmean(armaData, 8);
        //        kmean.Cluster(armaData, desiredClusterNum, assignments, centroids);
        //        centroids = centroids.t();
        //        Miscellaneous::Conversion::armamat2cvmat_<float, float>(centroids, classModelSet[i]);

        //        cv::TermCriteria critera;
        //        critera.type = cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS;
        //        critera.maxCount = 500;
        //        critera.epsilon = 0.1;
        //        cv::Mat bestLabels;
        //        cv::kmeans(materialTrainData[i], desiredClusterNum, bestLabels, critera, 20, cv::KMEANS_PP_CENTERS, classModelSet[i]);

        m_vecClassVecModelSet[i] = classModelSet[i];
        m_vecRgb2IrTransformSet[i] = cv::Mat_<float>(foundCluster, 3);
        std::vector<MaterialFeatureSet> materialModelSet;

        std::cout<<"material type "<<i<<std::endl;
        for(int j = 0; j < classModelSet[i].rows; ++j){
            //convert to MaterialFeatureSet format
            float *srcPtr = classModelSet[i].ptr<float>(j);
            MaterialFeatureSet feature;
            feature.vecMaterialFeatures.resize(4);

            //textonDictionary
            feature.textonHistogram = cv::Mat_<float>(1, ms_nGlobalTextonDictionarySize);
            float *dstPtr = feature.textonHistogram.ptr<float>(0);
            for(int k = 0; k < ms_nGlobalTextonDictionarySize; ++k){
                *dstPtr++ = *srcPtr++;
            }
            feature.vecMaterialFeatures[0] = feature.textonHistogram;

            //spectrum
            feature.spectrumResponse = cv::Mat_<float>(1, ms_nSpectrumNum);
            dstPtr = feature.spectrumResponse.ptr<float>(0);
            for(int k = 0; k < ms_nSpectrumNum; ++k){
                *dstPtr++ = *srcPtr++;
            }
            feature.vecMaterialFeatures[1] = feature.spectrumResponse;

            //ndvi
            feature.ndviHistogram = cv::Mat_<float>(1, ms_nNdviHistNum);
            dstPtr = feature.ndviHistogram.ptr<float>(0);
            for(int k = 0; k < ms_nNdviHistNum; ++k){
                *dstPtr++ = *srcPtr++;
            }
            feature.vecMaterialFeatures[2] = feature.ndviHistogram;

            //chroma
            feature.chromaHistogram = cv::Mat_<float>(1, ms_nChromaHistNum);
            dstPtr = feature.chromaHistogram.ptr<float>(0);
            for(int k = 0; k < ms_nChromaHistNum; ++k){
                *dstPtr++ = *srcPtr++;
            }
            feature.vecMaterialFeatures[3] = feature.chromaHistogram;

            //compute transform matrix
            cv::Mat N = cv::Mat_<float>(classCnt[i], 1);
            cv::Mat RGB = cv::Mat_<float>(classCnt[i], 3);
            int dstCnt = 0;
            for(size_t k = 0; k < classCnt[i]; ++k){
                //                if(assignments(k) != j){
                //                    continue;
                //                }
                if(bestLabels.at<int>(k) != j){
                    continue;
                }
                N.at<float>(dstCnt, 0) = materialTrainData[i].at<float>(k, ms_nGlobalTextonDictionarySize+3);
                RGB.at<float>(dstCnt, 0) = materialTrainData[i].at<float>(k, ms_nGlobalTextonDictionarySize);
                RGB.at<float>(dstCnt, 1) = materialTrainData[i].at<float>(k, ms_nGlobalTextonDictionarySize+1);
                RGB.at<float>(dstCnt, 2) = materialTrainData[i].at<float>(k, ms_nGlobalTextonDictionarySize+2);
                dstCnt++;
            }
            N.adjustROI(0, dstCnt-classCnt[i], 0, 0);
            RGB.adjustROI(0, dstCnt-classCnt[i], 0, 0);
            cv::Mat T = (RGB.t()*RGB).inv()*RGB.t()*N;
            for(int k = 0; k < 3; k++){
                m_vecRgb2IrTransformSet[i].at<float>(j, k) = T.at<float>(k,0);
            }
            std::cout<<"Transform matrix T\n"<<T<<std::endl;
            feature.rgb2irMat = T;
            //push model to buffer
            materialModelSet.push_back(feature);
        }
        m_vecClassMaterialModelSets[i] = materialModelSet;
        QString fileName = "model_"+QString::number(i);
        QString transformFileName = "T_"+QString::number(i);
        Miscellaneous::IO::data2Text_<float>(m_vecClassVecModelSet[i], fileName.toStdString());
        Miscellaneous::IO::data2Text_<float>(m_vecRgb2IrTransformSet[i], transformFileName.toStdString());
    }
    //test train result
    {
        double classErrorCnt[categoryNum][categoryNum];

        MaterialFeatureDist<float> dist;
        dist.setFeatureUtility(bUseTexton, bUseSpectrum, bUseNdvi, bUseChroma);
        for(int i = 0; i < categoryNum; ++i){
            classErrorCnt[i][0] = 0;
            classErrorCnt[i][1] = 0;
            classErrorCnt[i][2] = 0;
            classErrorCnt[i][3] = 0;
            classErrorCnt[i][4] = 0;
            int nDataCnt = materialTrainData[i].rows;
            for(int j = 0; j < nDataCnt; ++j){
                double dMinDist = std::numeric_limits<float>::max();
                int nMinDistClass = 0;
                for(int k = 0; k < m_vecClassVecModelSet.size(); ++k){
                    for(int l = 0; l < m_vecClassVecModelSet[k].rows; ++l){
                        float *modelPtr = (m_vecClassVecModelSet[k].row(l)).ptr<float>(0);
                        float *dataPtr = (materialTrainData[i].row(j)).ptr<float>(0);
                        double dDist = dist(modelPtr, dataPtr, 20);
                        if(dDist < dMinDist){
                            dMinDist = dDist;
                            nMinDistClass = k;
                        }
                    }
                }
                classErrorCnt[i][nMinDistClass]+= 1;
            }

            for(int j = 0; j < categoryNum; ++j){
                classErrorCnt[i][j] = classErrorCnt[i][j]/nDataCnt;
                std::cout<<classErrorCnt[i][j]<<" ";
            }
            std::cout<<std::endl;
        }
    }
}


void MultiMaterialClassifier::UnsupervisedModelBuilding(QString dataAddress)
{
    std::cout<<"unsupervised model building\n";
    std::cout.flush();
    QStringList imgDirList = Miscellaneous::loadImageList(dataAddress);
    if(imgDirList.isEmpty()){
        return;
    }

    int clusterNum = 60;
    int maxTrainDataSize = 20000;
    cv::Mat materialTrainData;
    int totalCnt = 0;

    SLICSuperpixel slic;
    XtionFull device;
    Rectification rectifier(device);

    for(int i = 0; i < imgDirList.size();){
        cv::Mat baseImgOrg, referImgOrg, baseDepthImgOrg, baseMaterialImgOrg;
        StereoData data;
        baseDepthImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data()); i++;
        referImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data()); i++;
        baseImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data());i++;
        baseMaterialImgOrg = cv::imread(imgDirList[i].toLocal8Bit().data(), cv::IMREAD_GRAYSCALE); i++;

        cv::Rect validRoiBase, validRoiRefer;
        rectifier.rectify(baseImgOrg, data.baseImg, validRoiBase, RECT_IMAGE_TYPE_RIGHT, cv::INTER_LANCZOS4);
        rectifier.rectify(referImgOrg, data.referImg, validRoiRefer, RECT_IMAGE_TYPE_LEFT, cv::INTER_LANCZOS4);
        rectifier.rectify(baseDepthImgOrg, data.baseDepth, validRoiBase, RECT_IMAGE_TYPE_RIGHT, cv::INTER_NEAREST);
        rectifier.rectify(baseMaterialImgOrg, data.baseMaterial, validRoiBase, RECT_IMAGE_TYPE_RIGHT, cv::INTER_NEAREST);
        Miscellaneous::Conversion::depth2horDisp_<ushort, float>(data.baseDepth, data.baseOrgDisp);

        slic.init(data.baseImg, 150, 80, 40);
        slic.generateSuperPixels();

        std::vector<MaterialFeatureSet> clusterFeatureSet;
        extractClusterFeature(data, slic, clusterFeatureSet);

        if(materialTrainData.empty()){
            int nTotalLength = 0;
            for(int j = 0; j < clusterFeatureSet[0].vecMaterialFeatures.size(); ++j){
                nTotalLength += clusterFeatureSet[0].vecMaterialFeatures[j].cols;
            }
            
            materialTrainData = cv::Mat_<float>(maxTrainDataSize, nTotalLength, 0.f);
        }

        std::vector<cv::Point2i> clusterCenters = slic.getClusterCenters();
        cv::Mat clusterIndexMap = slic.getClustersIndex();
        clusterNum = clusterCenters.size();

        for(int j = 0; j < clusterNum; ++j){
            cv::Mat mask;
            cv::Rect patchRoi;
            getClusterMask(clusterIndexMap, clusterCenters[j], j, mask, patchRoi);
            cv::Mat searchPatch = data.baseImg(patchRoi);

            QString patchName =  "/home/shenyunjun/Data/Output/Patch/"+QString::number(totalCnt)+".tiff";
            cv::imwrite(patchName.toLocal8Bit().data(), searchPatch);
            
            float *dstPtr = materialTrainData.ptr<float>(totalCnt);
            for(int k = 0; k < clusterFeatureSet[j].vecMaterialFeatures.size(); ++k){
                float *srcPtr = clusterFeatureSet[j].vecMaterialFeatures[k].ptr<float>(0);
                int featureLength = clusterFeatureSet[j].vecMaterialFeatures[k].cols;
                for(int l = 0; l < featureLength; ++l){
                    *dstPtr++ = *srcPtr++;
                }
            }
            totalCnt += 1;
        }
        slic.clear();
    }
    
    materialTrainData.adjustROI(0, totalCnt-maxTrainDataSize, 0, 0);
    //    std::cout<<materialTrainData;

    int desiredK = 20;
    cvflann::KMeansIndexParams kmeanParam(6, 100, cvflann::FLANN_CENTERS_RANDOM);
    cv::Mat globalModelSet = cv::Mat_<float>(desiredK, materialTrainData.cols);
    MaterialFeatureDist<float> dist;
    dist.setFeatureUtility(bUseTexton, bUseSpectrum,bUseNdvi, bUseChroma);
    int foundCluster = cv::flann::hierarchicalClustering<MaterialFeatureDist<float> >(materialTrainData,
                                                                                      globalModelSet,
                                                                                      kmeanParam,
                                                                                      dist);
    globalModelSet.adjustROI(0, foundCluster - desiredK, 0, 0);

    std::vector<MaterialFeatureSet> vecModelSet;
    //concatenated feature vector to struct format
    for(int j = 0; j < globalModelSet.rows; ++j){
        float *srcPtr = globalModelSet.ptr<float>(j);

        MaterialFeatureSet feature;
        feature.vecMaterialFeatures.resize(4);
        //textonDictionary
        feature.textonHistogram = cv::Mat_<float>(1, ms_nGlobalTextonDictionarySize);
        float *dstPtr = feature.textonHistogram.ptr<float>(0);
        for(int k = 0; k < ms_nGlobalTextonDictionarySize; ++k){
            *dstPtr++ = *srcPtr++;
        }
        feature.vecMaterialFeatures[0] = feature.textonHistogram;

        //spectrum
        feature.spectrumResponse = cv::Mat_<float>(1, ms_nSpectrumNum);
        dstPtr = feature.spectrumResponse.ptr<float>(0);
        for(int k = 0; k < ms_nSpectrumNum; ++k){
            *dstPtr++ = *srcPtr++;
        }
        feature.vecMaterialFeatures[1] = feature.spectrumResponse;

        //ndvi
        feature.ndviHistogram = cv::Mat_<float>(1, ms_nNdviHistNum);
        dstPtr = feature.ndviHistogram.ptr<float>(0);
        for(int k = 0; k < ms_nNdviHistNum; ++k){
            *dstPtr++ = *srcPtr++;
        }
        feature.vecMaterialFeatures[2] = feature.ndviHistogram;

        //chroma
        feature.chromaHistogram = cv::Mat_<float>(1, ms_nChromaHistNum);
        dstPtr = feature.chromaHistogram.ptr<float>(0);
        for(int k = 0; k < ms_nChromaHistNum; ++k){
            *dstPtr++ = *srcPtr++;
        }
        feature.vecMaterialFeatures[3] = feature.chromaHistogram;

        vecModelSet.push_back(feature);
        cv::Mat hist;
        Miscellaneous::plotHist(hist, feature.vecMaterialFeatures[0]);
        QString windName = QString::number(j);
        cv::imshow(windName.toLocal8Bit().data(), hist);
    }

    m_vecModelSets = vecModelSet;


    QString fileName = "self_cluster_model";
    Miscellaneous::IO::data2Text_<float>(globalModelSet, fileName.toStdString());

    //test error_in
    for(int i = 0; i < materialTrainData.rows; ++i){
        int closestModelIndex = 0;
        float minDist = std::numeric_limits<float>::max();
        for(int j = 0; j < globalModelSet.rows; ++j){
            float *testPtr = materialTrainData.ptr<float>(i);
            float *modelPtr = globalModelSet.ptr<float>(j);
            float distance = dist(testPtr, modelPtr, 20);
            if(distance < minDist){
                minDist = distance;
                closestModelIndex = j;
            }
        }
        QString patchName =  "/home/shenyunjun/Data/Output/Patch/"+QString::number(i)+".tiff";
        cv::Mat patch =  cv::imread(patchName.toLocal8Bit().data());
        QString patchOutputName = "/home/shenyunjun/Data/Output/SortedPatch/class_"+QString::number(closestModelIndex)+"_"+QString::number(i)+".tiff";
        cv::imwrite(patchOutputName.toLocal8Bit().data(), patch);
    }
}

void MultiMaterialClassifier::concatenateVector2Mat(std::vector<Mat> &dataVector, Mat &vectorMat)
{
    int nTotalLength = 0;
    for(int i = 0; i < dataVector.size(); ++i){
        nTotalLength += dataVector[i].cols;
    }

    vectorMat = cv::Mat_<float>(1, nTotalLength);
    float *dstPtr = vectorMat.ptr<float>(0);
    for(int i = 0; i < dataVector.size(); ++i){
        float *srcPtr = dataVector[i].ptr<float>(0);
        int nFeaLength = dataVector[i].cols;
        for(int j = 0; j < nFeaLength; ++j){
            *dstPtr++ = *srcPtr++;
        }
    }
}

void MultiMaterialClassifier::extractClusterFeature( StereoData &data,
                                                     SLICSuperpixel &slic,
                                                     std::vector<MaterialFeatureSet> &clusterFeatureSet)
{
    cv::Mat f_grayImg, hsvImg, rgbImg;
    if(data.baseImg.channels() == 1){
        cvtColor(data.baseImg, rgbImg, cv::COLOR_GRAY2BGR);
        cv::Mat grayImg = data.baseImg.clone();
        grayImg.convertTo(f_grayImg, CV_32F);
        return;
    }else{
        cv::Mat grayImg, f_baseImg;
        data.baseImg.convertTo(f_baseImg, CV_32FC3);
        cv::cvtColor(data.baseImg, grayImg, cv::COLOR_BGR2GRAY);

        cv::cvtColor(f_baseImg, hsvImg, cv::COLOR_BGR2HSV);
        rgbImg = data.baseImg.clone();
        grayImg.convertTo(f_grayImg, CV_32F);
    }
    bool bUseMaterailInfo = false;
    if(!data.baseMaterial.empty()){
        bUseMaterailInfo = true;
    }
    //    cv::Mat component[3];
    //    cv::split(hsvImg, component);
    //    cv::imshow("hue", component[0]/360);
    //    cv::waitKey(0);

    int n_row = data.baseImg.rows;
    int n_col = data.baseImg.cols;

    //    cv::flann::Index flann_index;
    //    flann_index.build(m_globalTextonDictionary, cv::flann::LinearIndexParams(),cvflann::FLANN_DIST_EUCLIDEAN);

    //    cv::Mat responseVec;
    //    cv::Mat responseVecIndices;
    cv::Mat responseMatIndices;
    {
        std::vector<cv::Mat> responseMap(m_nFilterKernelSetSize);
        cv::Mat totalResponseMat;
        for(int j = 0; j < m_nFilterKernelSetSize; ++j){
            cv::filter2D(f_grayImg, responseMap[j], CV_32F, m_vecFilterKernelSet[j]);
        }
        cv::merge(responseMap, totalResponseMat);
        //visualize response map
        //        for(int i = 0; i < responseMap.size(); ++i){
        //            QString fileName = QString::number(i)+"_response.tiff";
        //            cv::Mat visualImg = responseMap[i].clone();
        //            visualImg = cv::abs(visualImg);
        //            double min,max;
        //            cv::minMaxLoc(visualImg,&min, &max);
        //            cv::Mat saveImg;
        //            visualImg = 255*(visualImg - min)/(max-min);
        //            visualImg.convertTo(saveImg,CV_8U);
        //            cv::imwrite(fileName.toLocal8Bit().data(), saveImg);
        //        }
        //        responseVec = cv::Mat_<float>(totalResponseMat.rows*totalResponseMat.cols, totalResponseMat.channels());
        //        for(int j = 0; j < totalResponseMat.rows; ++j){
        //            float *srcPtr = totalResponseMat.ptr<float>(j);
        //            for(int k = 0; k < totalResponseMat.cols; ++k){
        //                float *dstPtr = responseVec.ptr<float>(j*n_col+k);
        //                for(int l = 0; l < totalResponseMat.channels(); ++l){
        //                    *dstPtr++ = *srcPtr++;
        //                }
        //            }
        //        }

        responseMatIndices = cv::Mat_<uchar>(responseMap[0].rows, responseMap[0].cols);
        for(int i = 0; i < responseMatIndices.rows; ++i){
            for(int k = 0; k < responseMatIndices.cols; ++k){
                float *responsePtr = totalResponseMat.ptr<float>(i)+k*m_nFilterKernelSetSize;
                cv::Mat testVector = cv::Mat_<float>(1, m_nFilterKernelSetSize, responsePtr);

                int nMinDistTextonIndex = 0;
                double dMinDist = std::numeric_limits<float>::max();
                for(int j = 0; j < m_globalTextonDictionary.rows; ++j){
                    double dist = cv::compareHist(testVector, m_globalTextonDictionary.row(j), cv::HISTCMP_CORREL);
                    if(dist < dMinDist){
                        nMinDistTextonIndex = j;
                        dMinDist = dist;
                    }
                }
                responseMatIndices.at<uchar>(i,k) = nMinDistTextonIndex;
            }
        }
        cv::imwrite("wordMap.tiff", responseMatIndices*5);
    }
    //    flann_index.knnSearch(responseVec, responseVecIndices, cv::noArray(), 1);

    //    responseVecIndices = cv::Mat_<int>(responseVec.rows, 1);
    //    for(int i = 0; i < responseVec.rows; ++i){
    //        int nMinDistTextonIndex = 0;
    //        double dMinDist = std::numeric_limits<float>::max();
    //        cv::Mat testVector = responseVec.row(i);
    //        for(int j = 0; j < m_globalTextonDictionary.rows; ++j){
    //            double dist = cv::compareHist(testVector, m_globalTextonDictionary.row(j), cv::HISTCMP_CORREL);
    //            if(dist < dMinDist){
    //                nMinDistTextonIndex = j;
    //                dMinDist = dist;
    //            }
    //        }
    //        responseVecIndices.at<int>(i,0) = nMinDistTextonIndex;
    //    }

    cv::Mat spectrumImg = cv::Mat_<cv::Vec4f>(n_row, n_col);
    {
        uchar *rgbPtr;
        float *spectrumPtr;
        float *dispPtr;
        for(int j = 0; j < n_row; ++j){
            rgbPtr = data.baseImg.ptr<uchar>(j);
            dispPtr= data.baseOrgDisp.ptr<float>(j);
            spectrumPtr = spectrumImg.ptr<float>(j);
            for(int k = 0; k < n_col; ++k){
                *spectrumPtr++ = static_cast<float>(*rgbPtr++);
                *spectrumPtr++ = static_cast<float>(*rgbPtr++);
                *spectrumPtr++ = static_cast<float>(*rgbPtr++);
                float disp = *dispPtr++;
                uchar material = 0;
                if(bUseMaterailInfo){
                    material = data.baseMaterial.at<uchar>(j,k);
                }
                if((disp == 0)&&(material != MATERIAL_SKY)){
                    disp = 1;
                }
                if((k+disp) > (n_col-1)){
                    *spectrumPtr++ = 0;
                }else{
                    *spectrumPtr++ = static_cast<float>((Miscellaneous::getValue_<uchar>(k+disp, j, 1, data.referImg)).at<uchar>(0,0));
                }
                //                if((disp == 0)&&(material != MATERIAL_SKY)){
                //                    *spectrumPtr++ = -1;
                //                }else{
                //                    *spectrumPtr++ = static_cast<float>((Miscellaneous::getValue_<uchar>(k+disp, j, 1, data.referImg)).at<uchar>(0,0));
                //                }
            }
        }
    }

    //for each cluster, compute feature
    std::vector<cv::Point2i> clusterCenters = slic.getClusterCenters();
    cv::Mat clusterIndexMap = slic.getClustersIndex();
    int clusterNum = clusterCenters.size();
    clusterFeatureSet.resize(clusterNum);
    for(int j = 0; j < clusterNum; ++j){
        MaterialFeatureSet featureSet;
        featureSet.vecMaterialFeatures.resize(4);

        cv::Mat mask;
        cv::Rect patchRoi;
        getClusterMask(clusterIndexMap, clusterCenters[j], j, mask, patchRoi);
        //        cv::imshow("mask",mask);
        //        cv::waitKey(0);
        int baseX = patchRoi.x;
        int baseY = patchRoi.y;
        int clusterCnt = 0;

        //voting for texton
        cv::Mat textonHistogram = cv::Mat_<float>(1, ms_nGlobalTextonDictionarySize, 0.0f);
        float *textonHistPtr = textonHistogram.ptr<float>(0);
        clusterCnt = 0;
        for(int k = 0; k < mask.rows; ++k){
            int y = k+baseY;
            for(int l = 0; l < mask.cols; ++l){
                uchar ucMaskValue = mask.at<uchar>(k, l);
                if(ucMaskValue < 200){
                    continue;
                }
                int x = l+baseX;
                //                int textonHistBin = responseVecIndices.at<int>(y*n_col+x);
                int textonHistBin = responseMatIndices.at<uchar>(y,x);
                textonHistPtr[textonHistBin] += 1;
                clusterCnt += 1;
            }
        }
        for(int k = 0; k < ms_nGlobalTextonDictionarySize; ++k){ //normalize
            textonHistPtr[k] = textonHistPtr[k]/clusterCnt;
        }
        featureSet.textonHistogram = textonHistogram;
        featureSet.vecMaterialFeatures[FEATURE_INDEX_TEXTON] = textonHistogram;
        //        std::cout<<"cluster "<<j<<"texton "<<textonHistogram<<std::endl;

        //spectrum response
        cv::Mat spectrumResponse = cv::Mat_<float>(1, ms_nSpectrumNum, 0.0f);
        {
            float *spectrumPtr;
            double spectrumSum[ms_nSpectrumNum];
            double totalSum = 0;
            for(int i = 0; i < 4; ++i){
                spectrumSum[i] = 0;
            }
            for(int k = 0; k < mask.rows; ++k){
                int y = k+baseY;
                spectrumPtr = spectrumImg.ptr<float>(y);
                for(int l = 0; l < mask.cols; ++l){
                    uchar ucMaskValue = mask.at<uchar>(k, l);
                    if(ucMaskValue < 200){
                        continue;
                    }
                    int shift = (l+baseX)*4;
                    if(spectrumPtr[shift+3] < 0){
                        continue;
                    }

                    spectrumSum[0] = spectrumPtr[shift];
                    spectrumSum[1] = spectrumPtr[shift+1];
                    spectrumSum[2] = spectrumPtr[shift+2];
                    spectrumSum[3] = spectrumPtr[shift+3];
                }
            }
            //static sum for normalization
            for(int k = 0; k < ms_nSpectrumNum; ++k){
                totalSum += spectrumSum[k];
            }
            if(totalSum == 0){
                spectrumResponse = cv::Mat_<float>(1, ms_nSpectrumNum, 0.25f);
            }else{
                for(int k = 0; k < ms_nSpectrumNum; ++k){
                    spectrumResponse.at<float>(0,k) = spectrumSum[k]/totalSum;
                }
            }
        }
        featureSet.spectrumResponse = spectrumResponse;
        featureSet.vecMaterialFeatures[FEATURE_INDEX_SPECTRUM] = spectrumResponse;

        //ndvi histogram
        cv::Mat ndviHistogram = cv::Mat_<float>(1, ms_nNdviHistNum, 0.0f);
        {
            int nPixelCnt = 0;
            float *spectrumPtr;
            double ndviHist[10];
            for(int i = 0; i < ms_nNdviHistNum; ++i){
                ndviHist[i] = 0;
            }
            float scale = 2.0/ms_nNdviHistNum + 0.0001;
            for(int k = 0; k < mask.rows; ++k){
                int y = k+baseY;
                spectrumPtr = spectrumImg.ptr<float>(y);
                for(int l = 0; l < mask.cols; ++l){
                    uchar ucMaskValue = mask.at<uchar>(k, l);
                    if(ucMaskValue < 200){
                        spectrumPtr += ms_nSpectrumNum;
                        continue;
                    }
                    int shift = (l+baseX)*4;
                    if(spectrumPtr[shift+3] < 0){
                        continue;
                    }
                    nPixelCnt++;
                    float ndvi = (spectrumPtr[shift+2]-spectrumPtr[shift+3])/(spectrumPtr[shift+2]+spectrumPtr[shift+3]+0.001);
                    int discreteNdviBin = static_cast<int>((ndvi+1)/scale);
                    ndviHist[discreteNdviBin] += 1;
                }
            }
            if(nPixelCnt == 0){
                ndviHistogram = cv::Mat_<float>(1, ms_nNdviHistNum, -1.0f);
            }else{
                for(int k = 0; k < ms_nNdviHistNum; ++k){
                    ndviHistogram.at<float>(0,k) = ndviHist[k]/nPixelCnt;
                }
            }
        }
        featureSet.ndviHistogram = ndviHistogram;
        featureSet.vecMaterialFeatures[2] = ndviHistogram;

        //chroma histogram
        cv::Mat chromaHistogram = cv::Mat_<float>(1, ms_nChromaHistNum, 0.0f);
        {
            double hsvHist[ms_nChromaHistNum];
            int nHBinSize = 9;
            int nSBinSize = 4;
            int nVBinSize = 1;
            int nHSVColorBinSize = nHBinSize*nSBinSize*nVBinSize;
            float fHScale = (360.0/nHBinSize)*1.001;
            float fSScale = (1.0/nSBinSize)*1.001;
            float fVScale = (256.0/nVBinSize)*1.001;
            float fGrayScale = (256/4)*1.001;
            for(int i = 0; i < ms_nChromaHistNum; ++i){
                hsvHist[i] = 0;
            }
            float *fpHsvImg;
            for(int k = 0; k < mask.rows; ++k){
                int y = k+baseY;
                fpHsvImg = hsvImg.ptr<float>(y);
                for(int l = 0; l < mask.cols; ++l){
                    uchar ucMaskValue = mask.at<uchar>(k, l);
                    if(ucMaskValue < 200){
                        continue;
                    }
                    int shift = (l+baseX)*3;
                    //                    std::cout<<fpHsvImg[0]<<" "<<fpHsvImg[1]<<" "<<fpHsvImg[2]<<std::endl;
                    //                    std::cout.flush();
                    if(fpHsvImg[shift+1] < 0.02){
                        int nBin = static_cast<int>(fpHsvImg[shift+2]/fGrayScale) + nHSVColorBinSize;
                        hsvHist[nBin] += 1;
                    }else{
                        int nHBin = static_cast<int>(fpHsvImg[shift]/fHScale);
                        int nSBin = static_cast<int>(fpHsvImg[shift+1]/fSScale);
                        int nVBin = static_cast<int>(fpHsvImg[shift+2]/fVScale);
                        int nBin = nHBinSize*(nSBin*nVBinSize + nVBin)+nHBin;
                        hsvHist[nBin] += 1;
                    }
                }
            }
            //normalize
            for(int k = 0; k < ms_nChromaHistNum; ++k){
                chromaHistogram.at<float>(0,k) = hsvHist[k]/clusterCnt;
            }
        }
        featureSet.chromaHistogram = chromaHistogram;
        featureSet.vecMaterialFeatures[3] = chromaHistogram;

        //        std::cout<<featureSet.vecMaterialFeatures[0]<<" "<<featureSet.vecMaterialFeatures[1]<<" "<<featureSet.vecMaterialFeatures[2]
        //                                                      <<" "<<featureSet.vecMaterialFeatures[3]<<std::endl;

        //put feature into feature vector
        clusterFeatureSet[j] = featureSet;
    }
}

void MultiMaterialClassifier::getClusterMask(Mat clusterIndexMap,
                                             cv::Point2i clusterCenter,
                                             int tgtClusterIndex,
                                             cv::Mat &mask,
                                             cv::Rect &roi)
{
    int nCol = clusterIndexMap.cols;
    int nRow = clusterIndexMap.rows;

    int centerX = clusterCenter.x;
    int centerY = clusterCenter.y;
    int baseX, baseY;
    {
        int ox = centerX - 60, oy = centerY - 60;
        if(ox < 0){
            ox = 0;
        }
        if(oy < 0){
            oy = 0;
        }

        int width = 120, height = 120;
        if((ox+width) > nCol){
            width = nCol - ox -1;
        }
        if((oy+height) > nRow){
            height = nRow - oy - 1;
        }

        roi.x = ox; roi.y = oy; roi.width  = width; roi.height = height;
        baseX = ox; baseY = oy;
    }
    cv::Mat searchPatch = clusterIndexMap(roi);

    //generate mask
    mask = cv::Mat_<uchar>(searchPatch.rows, searchPatch.cols, (uchar)0);
    {
        cv::Point2i lc(searchPatch.cols, searchPatch.rows);
        cv::Point2i rb(0,0);
        uchar *maskPtr;
        int *clusterIndexPtr;
        for(int k = 0; k < searchPatch.rows; ++k){
            clusterIndexPtr = searchPatch.ptr<int>(k);
            maskPtr = mask.ptr<uchar>(k);
            for(int l = 0; l < searchPatch.cols; ++l){
                int clusterIndex = *clusterIndexPtr;
                if(clusterIndex != tgtClusterIndex){
                    clusterIndexPtr++;
                    maskPtr++;
                    continue;
                }
                if(k < lc.y){
                    lc.y = k;
                }
                if(l < lc.x){
                    lc.x = l;
                }
                if(k > rb.y){
                    rb.y = k;
                }
                if(l > rb.x){
                    rb.x = l;
                }
                *maskPtr = 255;
                maskPtr++;
                clusterIndexPtr++;
            }
        }
        mask.adjustROI(-lc.y, rb.y - searchPatch.rows, -lc.x, rb.x - searchPatch.cols);
        baseX += lc.x;
        baseY += lc.y;
    }
    //    cv::imshow("mask_before", mask);
    roi.x = baseX;
    roi.y = baseY;
    roi.width = mask.cols;
    roi.height = mask.rows;

    cv::Mat tmp;
    Miscellaneous::Erosion(mask, tmp, cv::MORPH_ELLIPSE, 4);
    if(sum(tmp)[0] > 10){
        mask = tmp;
    }
    //    cv::imshow("mask_after", mask);
    //    cv::waitKey(0);
}

//int MultiMaterialClassifier::hierarchyKmean(arma::fmat &data, int maxClusterNum)
//{
//    int nRow = data.n_rows;
//    int nCol = data.n_cols;
//    int nDimension = nRow;
//    int nMaxClusterNum = maxClusterNum;

//    std::vector<double> alpha(nMaxClusterNum+1, 1);
//    alpha[2] = 1-3.0/(4*nDimension);
//    for(int i = 3; i < (nMaxClusterNum+1); ++i){
//        float ak = alpha[i-1]+(1-alpha[i-1])/6.0;
//        alpha.push_back(ak);
//    }

//    std::vector<double> fk(nMaxClusterNum+1, 1);

//    std::vector<double> compactnessList(nMaxClusterNum+1, 1);

//    arma::fmat centroids;
//    arma::Col<size_t> assignments;
//    typedef MaterialFeatureMetric MaterialMetric;
//    mlpack::kmeans::KMeans<MaterialMetric> kmean;
//    kmean.Cluster(data, 1, assignments, centroids);

//    double l2Dist = 0;
//    for(int i = 0; i < nCol; ++i){
//        double dist = MaterialMetric::Evaluate<arma::Col<float> >(data.col(i), centroids.col(assignments(i)));
//        arma::Col<float> diff = data.col(i) - centroids.col(assignments(i));
//        arma::fmat left = diff.t();
//        left= left*diff;
//        l2Dist += dist;
//    }

//    compactnessList[1]=(l2Dist);
//    std::cout<<"fk score\n";
//    std::cout<<fk[1]<<' '<<1<<";\n";
//    for(int i = 2; i <= nMaxClusterNum; ++i){
//        kmean.Cluster(data, i, assignments, centroids);
//        l2Dist = 0;
//        for(int i = 0; i < nCol; ++i){
//            double dist = MaterialMetric::Evaluate<arma::Col<float> >(data.col(i), centroids.col(assignments(i)));
//            arma::Col<float> diff = data.col(i) - centroids.col(assignments(i));
//            arma::fmat left = diff.t();
//            left= left*diff;
//            l2Dist += dist;
//        }
//        compactnessList[i] = l2Dist;
//        if(l2Dist == 0){
//            fk[i] = 1;
//        }else{
//            fk[i] = (compactnessList[i]/(alpha[i]*compactnessList[i-1]));
//        }
//        std::cout<<fk[i]<<' '<<i<<";\n";
//    }

//    double dMinFkValue = std::numeric_limits<float>::max();
//    int nOptimalClusterNum = 1;
//    for(int i = 4; i < fk.size(); ++i){
//        if(fk[i] < dMinFkValue){
//            dMinFkValue = fk[i];
//            nOptimalClusterNum = i;
//        }
//    }

//    std::cout<<"compactness:\n";
//    for(int i = 0; i < compactnessList.size(); ++i){
//        std::cout<<compactnessList[i]<<' ';
//    }
//    std::cout<<std::endl;
//    std::cout.flush();

//    return nOptimalClusterNum;
//}

int MultiMaterialClassifier::hierarchyKmean(Mat &data, int maxClusterNum)
{
    int nRow = data.rows;
    int nCol = data.cols;
    int nDimension = nRow;
    int nMaxClusterNum = maxClusterNum;

    std::vector<double> alpha(maxClusterNum+1, 1);
    alpha[1] = 1;
    alpha[2] = (1-3.0/(4*nDimension));
    for(int i = 3; i <= nMaxClusterNum; ++i){
        float ak = alpha[i-1]+(1-alpha[i-1])/6.0;
        alpha[i] = ak;
    }

    std::vector<double> fk(nMaxClusterNum+1, 1);
    fk[1] = 1;

    std::vector<double> compactnessList(nMaxClusterNum+1, -1);

    cv::TermCriteria critera;
    critera.type = cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS;
    critera.maxCount = 500;
    critera.epsilon = 0.1;
    cv::Mat centroid;
    cv::Mat assignments;

    compactnessList[1] = (cv::kmeans(data, 1, assignments, critera, 20, cv::KMEANS_PP_CENTERS, centroid));
    std::cout<<"fk score\n";
    for(int i = 2; i <= nMaxClusterNum; ++i){
        double compactness = cv::kmeans(data, i, assignments, critera, 20, cv::KMEANS_PP_CENTERS, centroid);
        compactnessList[i] = compactness;
        if(compactness == 0){
            fk[i] = 1;
        }else{
            fk[i] = (compactnessList[i]/(alpha[i]*compactnessList[i-1]));
        }
        std::cout<<fk[i]<<' '<<i<<";\n";
    }

    double dMinFk = std::numeric_limits<float>::max();
    int nOptimalClusterNum = 1;
    for(int i = 1; i < fk.size(); ++i){
        if(fk[i] < dMinFk){
            dMinFk = fk[i];
            nOptimalClusterNum = i;
        }
    }

    std::cout<<"\n compactness:\n";
    for(int i = 0; i < compactnessList.size(); ++i){
        std::cout<<i<<" "<<compactnessList[i]<<"\n";
    }

    std::cout<<"\n alpha: \n";
    for(int i = 0; i < alpha.size(); ++i){
        std::cout<<i<<" "<<alpha[i]<<"\n";
    }

    std::cout<<"optimal k value is "<<nOptimalClusterNum<<std::endl;
    std::cout.flush();

    return nOptimalClusterNum;
}


size_t MultiMaterialClassifier::ms_nSpectrumNum = 4;
size_t MultiMaterialClassifier::ms_nGlobalTextonDictionarySize = 32;
size_t MultiMaterialClassifier::ms_nNdviHistNum = 10;
size_t MultiMaterialClassifier::ms_nChromaHistNum = 9*4*1+4;
