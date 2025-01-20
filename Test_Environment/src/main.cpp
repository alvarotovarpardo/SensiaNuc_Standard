#include "SensiaNuc.h"

int main()
{   
    SENSIANuc *m_pNuc;
    std::string sDir = "C:/CODE/SensiaNuc_STL/Test_Environment";
    std::string configName = "01";
    const int height = 512;
    const int width = 640;
    uchar bytes = 16;

    m_pNuc = new SENSIANuc(configName, height, width, bytes, sDir);

    m_pNuc -> cleanAll();

    cv::Mat Offset = m_pNuc -> getOffset();
    m_pNuc -> setGainForcingNucValue(2750);
    cv::Mat Gain = m_pNuc -> getGain();

    m_pNuc -> saveImageMBP("", 666);


    return 0;
}