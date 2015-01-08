#ifndef INTERPOLATION_H
#define INTERPOLATION_H

namespace Miscellaneous{

class CubicInterpolator{
public:
    CubicInterpolator(){
        //f(dx) = b*B^-1*C*y
        //B^-1*C = [-0.5000    1.5000   -1.5000    0.5000
    //              1.0000   -2.5000    2.0000   -0.5000
    //              -0.5000   0        0.5000   0
    //              0        1.0000    0         0]
        BIC = (cv::Mat_<float>(4,4)<<-0.5000,  1.5000,   -1.5000,    0.5000,
                                      1.0000,   -2.5000,    2.0000,   -0.5000,
                                      -0.5000,   0,        0.5000,   0,
                                      0,        1.0000,    0,         0);
//        std::cout<<"BIC"<<BIC<<std::endl;

        fpBIC[0][0] = -0.5000; fpBIC[0][1] = 1.5000; fpBIC[0][2] = -1.5000; fpBIC[0][3] = 0.5000;
        fpBIC[1][0] = 1.0000; fpBIC[1][1] = -2.5000; fpBIC[1][2] = 2.0000; fpBIC[1][3] = -0.5000;
        fpBIC[2][0] = -0.5000; fpBIC[2][1] = 0; fpBIC[2][2] = 0.5000; fpBIC[2][3] = 0;
        fpBIC[3][0] = 0; fpBIC[3][1] = 1.0000; fpBIC[3][2] = 0; fpBIC[3][3] = 0;

        BICY = BIC.clone();
        baseX = -1;
        baseY = -1;
    }

    ~CubicInterpolator(){

    }

    template<typename T>
    cv::Mat interpolate_(float x, int y, size_t length, cv::Mat &data){
        cv::Mat hist = cv::Mat_<T>(1, length);
        int fx = static_cast<int>(floor(x));
        float dx = x-fx;

        cv::Mat b = (cv::Mat_<T>(1,4)<<(dx*dx*dx), (dx*dx), dx, 1);
        cv::Mat Y = cv::Mat_<T>(4, length);
        {
            T *srcPtr, *dstPtr;
            srcPtr = data.ptr<T>(y);
            srcPtr += length*(fx-1);
            for(int i = 0; i < 4; ++i){
                dstPtr = Y.ptr<T>(i);
                for(int j = 0; j < length; ++j){
                   *dstPtr++ = *srcPtr++ ;
                }
            }
        }
        hist = b*BIC*Y;

        return hist;

    }

    template<typename T>
    T lineInterpolation_(float &x, T *data, size_t length){
        T interValue = 0;
        if(x < 1.0f){
            int floor = static_cast<int>(x);
            T dx = x - (T)floor;
            T y[4] = {data[0], data[0], data[1], data[2]};
            T b[4] = {dx*dx*dx, dx*dx, dx, 1};

            for(int i = 0; i < 4; ++i){
                interValue = interValue + (b[0]*fpBIC[0][i] + b[1]*fpBIC[1][i] + b[2]*fpBIC[2][i] + b[3]*fpBIC[3][i])*y[i];
            }
        }else if(x > (length-2)){
            int floor = static_cast<int>(x);
            T dx = x - (T)floor;
            T y[4] = {data[length-3], data[length-2], data[length-1], data[length-1]};
            T b[4] = {dx*dx*dx, dx*dx, dx, 1};

            for(int i = 0; i < 4; ++i){
                interValue = interValue + (b[0]*fpBIC[0][i] + b[1]*fpBIC[1][i] + b[2]*fpBIC[2][i] + b[3]*fpBIC[3][i])*y[i];
            }
        }else{
            int floor = static_cast<int>(x);
            T dx = x - (T)floor;
            T y[4] = {data[floor-1], data[floor], data[floor+1], data[floor+2]};
            T b[4] = {dx*dx*dx, dx*dx, dx, 1};

            for(int i = 0; i < 4; ++i){
                interValue = interValue + (b[0]*fpBIC[0][i] + b[1]*fpBIC[1][i] + b[2]*fpBIC[2][i] + b[3]*fpBIC[3][i])*y[i];
            }
        }
        return interValue;
    }

    template<typename T>
    cv::Mat continuousInterpolate_(float x, int y, size_t length, cv::Mat &data){
        cv::Mat hist = cv::Mat_<T>(1, length);
        int fx = static_cast<int>(x);
        float dx = x-fx;
        cv::Mat b = (cv::Mat_<T>(1,4)<<(dx*dx*dx), (dx*dx), dx, 1);

        if((fx == baseX)||(y == baseY)){    //if share same base point
            hist = b*BICY;
        }else{
            cv::Mat Y = cv::Mat_<T>(4, length);
            {
                T *srcPtr, *dstPtr;
                srcPtr = data.ptr<T>(y);
                srcPtr += length*(fx-1);
                for(int i = 0; i < 4; ++i){
                    dstPtr = Y.ptr<T>(i);
                    for(int j = 0; j < length; ++j){
                       *dstPtr++ = *srcPtr++ ;
                    }
                }
            }
            this->BICY = BIC*Y;
            baseX = fx, baseY = y;
            hist = b*BICY;
        }

        return hist;
    }

private:
    cv::Mat BIC;
    cv::Mat BICY;

    float fpBIC[4][4];
    int baseX, baseY;
};


template<typename T>
inline cv::Mat getValue_(float x, float y, size_t length, const cv::Mat &data)
{
    cv::Mat hist = cv::Mat_<T>(1,length);

    // relative indices
    int fx = static_cast<int>(floor(x));
    int fy = static_cast<int>(floor(y));
    int cx = static_cast<int>(ceil(x));
    int cy = static_cast<int>(ceil(y));
    // fractional part
    float ty = y - fy;
    float tx = x - fx;
    // set interpolation weights
    float w1 = (1 - tx) * (1 - ty);
    float w2 =      tx  * (1 - ty);
    float w3 = (1 - tx) *      ty;
    float w4 =      tx  *      ty;

    {
        const T *srcPtr;
        T *dstPtr;
        srcPtr = &data.ptr<T>(fy)[fx*length];
        dstPtr = hist.ptr<T>(0);
        for(int i = 0; i < length; ++i){
            *dstPtr++ = w1*srcPtr[i];
        }
    }

    {
        const T *srcPtr;
        T *dstPtr;
        srcPtr = &data.ptr<T>(fy)[cx*length];
        dstPtr = hist.ptr<T>(0);
        for(int i = 0; i < length; ++i){
            *dstPtr++ += w2*(*srcPtr++);
        }
    }

    {
        const T *srcPtr;
        T *dstPtr;
        srcPtr = &data.ptr<T>(cy)[fx*length];
        dstPtr = hist.ptr<T>(0);
        for(int i = 0; i < length; ++i){
            *dstPtr++ += w3*(*srcPtr++);
        }
    }

    {
        const T *srcPtr;
        T *dstPtr;
        srcPtr = &data.ptr<T>(cy)[cx*length];
        dstPtr = hist.ptr<T>(0);
        for(int i = 0; i < length; ++i){
            *dstPtr++ += w4*(*srcPtr++);
        }
    }
    return hist;
}

}

#endif // INTERPOLATION_H
