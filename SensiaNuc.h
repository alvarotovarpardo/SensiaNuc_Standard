#ifndef SENSIANUC_H
#define SENSIANUC_H

#include <chrono>

// For STL compatibility
#include <fstream>
#include <string>
#include <filesystem>
#include <regex>
#include <stdexcept>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// #include <opencv2/cudaarithm.hpp>

// #include <npp.h>

// #include "util/file_log.h"
// #include "sensia-nuc_global.h"
// #include "../config_camera/config_camera.h"

#include "pixel_manager.h"

extern "C" void gpuRefFria(unsigned short *pOut, int stepOut, unsigned short *pIn, int stepIn, unsigned short *pFria, int stepFria, int iWidth, int iHeight, int iMediaFria);
extern "C" void gpuCopy(unsigned short *pOut, int stepOut, unsigned short *pIn, int stepIn, int iWidth, int iHeight);
extern "C" void gpuPixelesMalos(unsigned short *pInOut, int stepOut, unsigned short *pPixelesMalos, int stepPixelexMalos, unsigned short *pMediana, int stepMedia, int iWidth, int iHeight);
extern "C" void gpuRepeatEdgeRightBottom(unsigned short *pInOut, int stepInOut, int iWidth, int iHeight);
extern "C" void gpuExpandPreviosLastEdge(unsigned short *pInOut, int stepInOut, int iWidth, int iHeight);

// Structure for defining points nuc
struct PointsNuc
{
    enum Points{
        NONE = 0,
        ONE = 1,
        TWO = 2
    };
};

typedef PointsNuc::Points EnumPointsNuc;


class SENSIANuc
{
public:
    SENSIANuc(std::string &sConfigName, const int &height, const int &width, uchar dataBitsInput, std::string sDir = "");
    ~SENSIANuc();

    enum EnumUniformizationMode {NORM_USUAL, NORM_PICO};
    cv::Mat getGain()       const;
    cv::Mat getOffset()     const;
    uint getFrameCount(bool trueMeansHotBuffer = false)     const;
    void saveImageMBP(std::string sPatCalibration = "", int iNuc = 20);
    void cleanAll();
    void calculateNUC1P(cv::Mat *Mat_raw, uint bufferSizeMax);
    void calculateNUC2P(cv::Mat *Mat_raw, uint bufferSizeMax, const bool &refCold, const bool &refHot);
    bool isNucDone();
    void endNuc();
    void getNucMat(cv::Mat *matOut, cv::Mat *matIn);
    void setGainForcingNucValue(int iValue);
    void setOffsetForcingNucValue(double bolometerValue, int iValue);
    void setAdjustMatrixForcingNucValue(int iValue);
    void calculateAdjustMatrix(cv::Mat *inputMat, uint bufferSizeMax, uint tint);


private:
    void init();
    std::mutex mutex;
    std::string  m_sPathNuc;
    std::string m_sPathNucStd;
    void calculateGainAndOffset();
    EnumUniformizationMode m_uniformizationMode;
    void calculateOffset(cv::Mat& coldReference, float coldAvg);
    void calculateGain(cv::Mat& coldRef, cv::Mat& hotRef, float coldAvg, float hotAvg);
    void getAdjustMatrixFromReference(cv::Mat& coldReference, float coldAverage);
    void performNUC1P();
    void saveMat(std::string filename, cv::Mat &Mat, int iSize);
    std::string getDirNuc(std::string sPatCalibration, int iNuc);
    void insertReferenceToBuffer(cv::Mat *mat, bool trueMeansHot);
    void applyNucUsual(cv::Mat &matIn, cv::Mat &matOut);
    void applyNucPico(cv::Mat &matIn, cv::Mat &matOut);
    void initGain();
    // void readOffsetFileBandsLimits(std::string fileOffsetU, float& down, float& up);
    // bool readOffsetFileBands(std::string offsetFilePath, uint bands, double bolometerValue);
    uint getCurrentOffsetBand(double bolometerValue, int bands,float fDownLimit, float fUpLimit);
    void initOffset();
    void initBadPix();
    void clearBuffers();
    bool readNucMat(std::string filepath, cv::Mat& mat, int size);
    void initAdjustMatrix();
    cv::Mat getAverageOfBuffer(std::vector<cv::Mat>& buffer);
    cv::Mat elementWiseMatMult(cv::Mat& a, cv::Mat& b, int t);
    std::vector<std::string> findFiles(const std::string& path, const std::regex& regex);
protected:
        int count;
        float Media_FRIA;
        bool nucDone;
        uchar dataBits;
        ushort maxDigitalLevelsValue;

        std::vector<cv::Mat> MatBuffer;
        std::vector<float>   MediasBuffer;
        std::vector<cv::Mat> MatBuffer_FRIA;
        std::vector<float>   MediasBuffer_FRIA;
        std::vector<cv::Mat> MatBuffer_CAL;
        std::vector<float>   MediasBuffer_CAL;

        std::vector<float>   Medias_ExtraRef;
        std::vector<float>   STD_ExtraRef;

    std::vector<cv::Mat> coldMatBuffer;
    std::vector<cv::Mat> hotMatBuffer;

        cv::Mat Reference_FRIA;
    cv::Mat Gain;
    cv::Mat Offset;
        cv::Mat GainU;
        cv::Mat OffsetU;
    cv::Mat AdjustMatrixU;

        float gain_up_limit;
        float gain_down_limit;
    int _height;
    int _width;
        int _size;
        int _sizeBytes;

        unsigned short* pixeles_malos;
    cv::Mat mat_pixeles_malos;
        cv::Mat mat_pixeles_malos_file;
        cv::Mat mat_mediana;

};

#endif // SENSIANUC_H