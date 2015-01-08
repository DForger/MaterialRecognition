#ifndef MULTIMATERIAL_CLASSIFIER_H
#define MULTIMATERIAL_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <QtCore>
//#include <mlpack/core.hpp>
//#include <mlpack/methods/kmeans/kmeans.hpp>

#include "Preprocessing/SLICSuperpixel.h"
#include "Miscellaneous/miscellaneous.h"
#include "opencv2/flann/dist.h"


using namespace Stereo;

enum{
    FEATURE_INDEX_TEXTON = 0,
    FEATURE_INDEX_SPECTRUM = 1,
    FEATURE_INDEX_MEAN_VALUE = 2,
    FEATURE_INDEX_CHROMA = 3
};



struct MaterialFeatureSet{
    std::vector<cv::Mat> vecMaterialFeatures;
    cv::Mat textonHistogram;
    cv::Mat spectrumResponse;
    cv::Mat ndviHistogram;
    cv::Mat chromaHistogram;
    cv::Mat colorCovMatrix;
    cv::Mat rgb2irMat;

    std::ostream &operator<<(std::ostream &out){
        for(int i = 0; i < this->vecMaterialFeatures.size(); ++i){
            float *dataPtr = this->vecMaterialFeatures[i].ptr<float>(0);
            out<<"Feature Vector "<<i<<" :";
            for(int j = 0; j < this->vecMaterialFeatures[i].cols; ++j){
                out<<(*dataPtr++)<<' ';
            }
            out<<std::endl;
        }
    }
};

class MultiMaterialClassifier
{
public:
    MultiMaterialClassifier();

    void smoothMaterialMap(cv::Mat &inputMap, cv::Mat &outputMap);

    void visualizeMaterialMap(cv::Mat &materialMap, cv::Mat &visualImg);

    void train(QString dataAddress);

    void load();

    void test(StereoData &data);

    //getter and setter
    void getFinalMaterialMap(cv::Mat &materialMap){
        materialMap = m_finalMaterialMap.clone();
    }

    void getVisualMaterialMap(cv::Mat &visualMap){
        visualMap = m_visualMaterialMap.clone();
    }

    void getColorBar(cv::Mat &colorBar);

    void extractClusterFeature(StereoData &data,
                               SLICSuperpixel &slic,
                               std::vector<MaterialFeatureSet> &clusterFeatureSet);

private:
    void buildFilterKernelSet(QString dataAddress);

    void buildGaborFilterKernelSet();

    void buildTextonDictionarySet(QString dataAddress);

    void buildGlobalTextonDictionary(QString dataAddress);

    void buildColorModelSet(QString dataAddress);

    void buildModelSet(QString dataAddress);
    
    void UnsupervisedModelBuilding(QString dataAddress);

    void concatenateVector2Mat(std::vector<cv::Mat> &dataVector, cv::Mat &vectorMat);

    void getClusterMask(cv::Mat clusterIndexMap,
                        cv::Point2i clusterCenter,
                        int clusterIndex,
                        cv::Mat &mask,
                        cv::Rect &roi);

//    int hierarchyKmean(arma::fmat &data, int maxClusterNum);

    int hierarchyKmean(cv::Mat &data, int maxClusterNum);

    void setFeatureUtility(bool useTexton, bool useChroma, bool useNdvi, bool useSpectrum){
        this->bUseChroma = useChroma;
        this->bUseNdvi = useNdvi;
        this->bUseSpectrum = useSpectrum;
        this->bUseTexton = useTexton;
    }

private:

    bool bUseTexton;
    bool bUseChroma;
    bool bUseNdvi;
    bool bUseSpectrum;

    size_t m_nMaterialTypes;

    size_t m_nFilterKernelSize;    //kernel size MxN
    size_t m_nFilterKernelSetSize; //how many kernel we use
    std::vector<cv::Mat> m_vecFilterKernelSet;

    cv::Mat m_globalTextonDictionary;

    std::vector<size_t> m_vecTextonDictionarySize;
    std::vector<cv::Mat> m_vecTextonDictionarySet;

    static size_t ms_nGlobalTextonDictionarySize;
    static size_t ms_nSpectrumNum;
    static size_t ms_nNdviHistNum;
    static size_t ms_nChromaHistNum;

    std::vector<size_t>  m_vecClassModelSize;
    std::vector<cv::Mat> m_vecClassVecModelSet;
    std::vector<cv::Mat> m_vecRgb2IrTransformSet;
    std::vector<std::vector<MaterialFeatureSet> > m_vecClassMaterialModelSets;
    std::vector<MaterialFeatureSet> m_vecModelSets;

    std::map<int, cv::Vec3b> m_tableMaterialColor;
    std::map<int, cv::String> m_tableMaterialName;

    cv::Mat m_finalMaterialMap;
    cv::Mat m_visualMaterialMap;

    friend class MaterialFeatureMetric;
    template<typename T> friend struct MaterialFeatureDist;
};


class MaterialFeatureMetric
{
public:
    /***
   * Default constructor does nothing, but is required to satisfy the Kernel
   * policy.
   */
    MaterialFeatureMetric() { }
    /**
   * Computes the distance between two points.
   */
    template<typename VecType>
    static double Evaluate(const VecType& a, const VecType& b){
        double dColorDist = 0.1;
        double dTextonDist = 0.1;
        double dNdviDist = 0.1;
        double dChromaDist = 0.1;

        int textonDictSize = MultiMaterialClassifier::ms_nGlobalTextonDictionarySize;
        int spectrumSize = MultiMaterialClassifier::ms_nSpectrumNum;
        int ndviHistSize = MultiMaterialClassifier::ms_nNdviHistNum;
        int chromaHistSize = MultiMaterialClassifier::ms_nChromaHistNum;

        int base = 0;
        dTextonDist += BhattacharyyaDist<VecType>(a, b, cv::Range(base, base+dTextonDist));

        base += textonDictSize;
        dColorDist += BhattacharyyaDist<VecType>(a, b, cv::Range(base, base+spectrumSize));

        base += spectrumSize;
        dNdviDist += BhattacharyyaDist<VecType>(a, b, cv::Range(base, base+ndviHistSize));

        base += ndviHistSize;
        dChromaDist += BhattacharyyaDist<VecType>(a, b, cv::Range(base, base+chromaHistSize));

        return dTextonDist*spectrumSize*10;
    }

    static double Compare(MaterialFeatureSet& a, MaterialFeatureSet& b){
        double textonSimilarity = 1- cv::compareHist(a.textonHistogram, b.textonHistogram, cv::HISTCMP_BHATTACHARYYA);
        double spectrumSimilarity = 1 - cv::compareHist(a.spectrumResponse, b.spectrumResponse, cv::HISTCMP_BHATTACHARYYA);
        double ndviSimilarity = 1 - cv::compareHist(a.ndviHistogram, b.ndviHistogram, cv::HISTCMP_BHATTACHARYYA);
        double chromaSimilarity = 1 - cv::compareHist(a.chromaHistogram, b.chromaHistogram, cv::HISTCMP_BHATTACHARYYA);

        return textonSimilarity*spectrumSimilarity*chromaSimilarity*10;
    }
    
    template<typename VecType>
    double static BhattacharyyaDist(const VecType& a, const VecType& b, cv::Range range){
        double sum = 0;
        int size = range.end - range.start;
        double mean_a = 0;
        double mean_b = 0;
        for(int i = range.start; i < range.end; ++i){
            sum += std::sqrt(a(i)*b(i));
            mean_a += a(i);
            mean_b += b(i);
        }
        mean_a = mean_a/size;
        mean_b = mean_b/size;

        return std::sqrt(1 - sum/(std::sqrt(mean_a*mean_b)*size+0.01));
    }
};

template<class T>
struct MaterialFeatureDist
{
    typedef cvflann::False is_kdtree_distance;
    typedef cvflann::True is_vector_space_distance;

    typedef T ElementType;
    typedef typename Accumulator<T>::Type ResultType;

    MaterialFeatureDist()
        : mbUseTexton(true),
          mbUseSpectrum(true),
          mbUseNdviHist(true),
          mbUseChromaHist(true)
    {
        textonDictSize = MultiMaterialClassifier::ms_nGlobalTextonDictionarySize;
        spectrumSize = MultiMaterialClassifier::ms_nSpectrumNum;
        ndviHistSize = MultiMaterialClassifier::ms_nNdviHistNum;
        chromaHistSize = MultiMaterialClassifier::ms_nChromaHistNum;
    }

    MaterialFeatureDist(int textonDictSize,
                        int spectrumSize,
                        int ndviHistSize,
                        int chromaHistSize)
        : mbUseTexton(true),
          mbUseSpectrum(true),
          mbUseNdviHist(true),
          mbUseChromaHist(true)
    {
        this->textonDictSize = textonDictSize;
        this->spectrumSize = spectrumSize;
        this->ndviHistSize = ndviHistSize;
        this->chromaHistSize = chromaHistSize;
    }

    template <typename Iterator1, typename Iterator2>
    ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType /*worst_dist*/ = -1) const
    {
        ResultType dSpectrumDist = 0.1;
        ResultType dTextonDist = 0.1;
        ResultType dNdviDist = 0.1;
        ResultType dChromaDist = 0.1;

        ResultType result = ResultType();

        int base = 0;
        dTextonDist += BhattacharyyaDist<Iterator1, Iterator2>(a, b, cv::Range(base, base+textonDictSize));

        base += textonDictSize;
        if(mbUseSpectrum){
            if((a[base] < 0)||(b[base] < 0)){
                dSpectrumDist = dTextonDist;
            }else{
                dSpectrumDist += BhattacharyyaDist<Iterator1, Iterator2>(a, b, cv::Range(base, base+spectrumSize));
            }
        }else{
            dSpectrumDist = 1;
        }

        base += spectrumSize;
        if(mbUseNdviHist){
            if((a[base] < 0)||(b[base] < 0)){
                dNdviDist = dTextonDist;
            }else{
                dNdviDist += BhattacharyyaDist<Iterator1, Iterator2>(a, b, cv::Range(base, base+ndviHistSize));
            }
        }else{
            dNdviDist = 1;
        }

        base += ndviHistSize;
        if(mbUseChromaHist){
            if((a[base] < 0)||(b[base] < 0)){
                dChromaDist = dTextonDist;
            }else{
                dChromaDist += BhattacharyyaDist<Iterator1, Iterator2>(a, b, cv::Range(base, base+chromaHistSize));
            }
        }else{
            dChromaDist = 1;
        }

        result = dTextonDist*dSpectrumDist*dNdviDist*dChromaDist*1000;
//        result = (dTextonDist+dSpectrumDist+dNdviDist+dChromaDist)*10;
        return result;
    }

    template <typename Iterator1, typename Iterator2>
    ResultType static BhattacharyyaDist(Iterator1& a, Iterator2& b, cv::Range range){
        ResultType sum = 0;
        int size = range.end - range.start;
        ResultType mean_a = 0;
        ResultType mean_b = 0;
        for(int i = range.start; i < range.end; ++i){
            sum = sum + std::sqrt(a[i]*b[i]);
            mean_a = mean_a + a[i];
            mean_b = mean_b + b[i];
        }
        mean_a = mean_a/size;
        mean_b = mean_b/size;

        return (ResultType)std::sqrt(1 - sum/(std::sqrt(mean_a*mean_b)*size+0.001));
    }

    void setFeatureUtility(bool bUseTexton, bool bUseSpectrum, bool bUseNdviHist, bool bUseChromaHist){
        mbUseTexton = bUseTexton;
        mbUseSpectrum = bUseSpectrum;
        mbUseNdviHist = bUseNdviHist;
        mbUseChromaHist = bUseChromaHist;
    }

private:
    int textonDictSize;
    int spectrumSize;
    int ndviHistSize;
    int chromaHistSize;

    bool mbUseTexton;
    bool mbUseSpectrum;
    bool mbUseNdviHist;
    bool mbUseChromaHist;
};
#endif // MULTIMATERIAL_CLASSIFIER_H
