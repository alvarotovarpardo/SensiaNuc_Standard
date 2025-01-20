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

    // m_pNuc -> setGainForcingNucValue(210);
    // m_pNuc -> setOffsetForcingNucValue(0, 210);

    // Gain = m_pNuc -> getGain();
    // Offset = m_pNuc -> getOffset();

    // const uchar* dataGain = Gain.ptr<uchar>();
    // for (int i = 0; i < std::min(100, static_cast<int>(Gain.total() * Gain.channels())); ++i) {
    //     std::cout << static_cast<int>(dataGain[i]) << " ";
    // }
    // std::cout << "\n";
    return 0;
}