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

    m_pNuc -> setGainForcingNucValue(2750);
    m_pNuc -> setOffsetForcingNucValue(0, 2750);

    m_pNuc -> cleanAll();
    cv::Mat Gain = m_pNuc -> getGain();
    cv::Mat Offset = m_pNuc -> getOffset();

    const float* dataGain = Gain.ptr<float>();
    std::cout << "Gain:\n\n";
    for (int i = 0; i < 50; ++i) {
        std::cout << static_cast<float>(dataGain[i]) << " ";
    }
    std::cout << "\n\n\n";

    std::cout << "Offset:\n\n";
    const float* dataOffset = Offset.ptr<float>();
    for (int i = 0; i < 50; ++i) {
        std::cout << static_cast<float>(dataOffset[i]) << " ";
    }

    cv::Mat seeGain;
    Gain.convertTo(seeGain, CV_8U, 255.0);
    cv::imshow("Gain", seeGain);


    cv::Mat seeOffset;
    Offset.convertTo(seeOffset, CV_8U, 255.0);
    cv::imshow("Offset", seeOffset);
    cv::waitKey(0);

    return 0;
}