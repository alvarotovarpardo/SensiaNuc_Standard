#include "SensiaNuc.h"

//// \brief Class for managing image Non-Uniformity Correction
////
//// \param sConfigName Camera configuration identifier. Will be "01" or similar by default.
//// \param height Camera height
//// \param width Camera width
//// \param sDir Test directory in the case of Analytics software. Leave empty otherwise.
SENSIANuc::SENSIANuc(std::string &sConfigName, const int &height, const int &width, uchar dataBitsInput, std::string sDir = "")
{
    _height = height;
    _width = width;
    m_uniformizationMode = SENSIANuc::NORM_USUAL;
    m_sPathNucStd = sDir + "/Calibration/" + sConfigName + "/Nucs";
    init();
    nucDone = false;
    dataBits = dataBitsInput;
    maxDigitalLevelsValue = (ushort) std::pow(2,static_cast<int>(dataBits)) - 1;
} 

SENSIANuc::~SENSIANuc()
{

}

/// \brief Initialize necessary objects for NUC processing
void SENSIANuc::init()
{
    Gain = cv::Mat::ones(_height,_width,CV_32F);
    Offset = cv::Mat::zeros(_height,_width,CV_32F);
    
    if(m_uniformizationMode == NORM_PICO){
        AdjustMatrixU = cv::Mat::ones(_height, _width, CV_32F);
    }

}



//// \brief get average matrix from a vector of matrices
///
/// \param inputBuffer Input buffer (vector) of matrices
/// \return average of matrix buffer
cv::Mat SENSIANuc::getAverageOfBuffer(std::vector<cv::Mat>& inputBuffer)
{
    if(inputBuffer.empty()) return cv::Mat();
    cv::Mat mat = cv::Mat::zeros(inputBuffer.front().rows,inputBuffer.front().cols,CV_32F);
    float size = static_cast<float>(inputBuffer.size());
    for(auto x : inputBuffer)
        mat += x/size;
    return mat;
}



/// \return Current Gain matrix
cv::Mat SENSIANuc::getGain()       const {return this->Gain;}


/// @return Current Offset matrix
cv::Mat SENSIANuc::getOffset()     const {return this->Offset;}

//// \brief Get count of frames in a specific buffer.
//// 
//// \param trueMeansHotBuffer True if the target is the hot reference buffer. False in the cold case.
//// \return Number of frames already taken in NUC processing
uint SENSIANuc::getFrameCount(bool trueMeansHotBuffer) const
{
    uint coldCount = coldMatBuffer.size(), hotCount = hotMatBuffer.size();
    if(!trueMeansHotBuffer)
        return coldCount;
    else
        return hotCount;
}


//// \brief Save matrix to disk
//// \param mat Matrix to save
//// \param filename Filename of the file to save the matrix to
//// \param size depth in bytes of the matrix to save (2 for unsigned short, 4 for float).
void SENSIANuc::saveMat(std::string filename, cv::Mat &mat, int size)
{
    std::ofstream file(filename, std::ios::binary);
    // Check
    if (!file.is_open()) {
        throw std::ios_base::failure("No se pudo abrir el archivo.");
    }
    //
    mutex.lock();
    file.write(reinterpret_cast<const char*>(mat.data), mat.rows * mat.cols * size);
    mutex.unlock();
}


//// \brief Get the NUC folder path. If it does not exist, the path is created.
//// \param sPathCalibration Path to Calibration folder.
//// \param iNuc Integration time (in us) for the current normalization.
std::string SENSIANuc::getDirNuc(std::string sPathCalibration, int iNuc)
{
    if(sPathCalibration == "")
    { 
        sPathCalibration = m_sPathNucStd;
    }
    std::filesystem::path dirCalibration = sPathCalibration;
    
    if (!std::filesystem::exists(dirCalibration)) 
    {
        std::filesystem::create_directories(dirCalibration);
    }

    std::filesystem::path dirNuc = dirCalibration / std::to_string(iNuc);

    if (!std::filesystem::exists(dirNuc)) 
    {
        std::filesystem::create_directory(dirNuc);
    }
    
    return dirNuc.string();
}

//// \brief Save Normalization matrices to disk.
//// \param sPatCalibration Path to Calibration folder.
//// \param iNuc Integration time (in us) for the current normalization.
void SENSIANuc::saveImageMBP(std::string sPatCalibration, int iNuc)
{
    std::string dirNuc = getDirNuc(sPatCalibration, iNuc);
    saveMat(dirNuc + "/GainU.raw", Gain, 4);
    saveMat(dirNuc + "/OffsetU.raw", Offset, 4);
    saveMat(dirNuc + "/bad_pix.raw", mat_pixeles_malos, 2);
}


/// @brief Clears all matrices and buffers
void SENSIANuc::cleanAll()
{
    initGain();
    initOffset();
    initBadPix(); // To be deprecated in favor of Pixman
    clearBuffers();
}


/// @brief Clears all buffers
void SENSIANuc::clearBuffers()
{
    coldMatBuffer.clear();
    hotMatBuffer.clear();
}


//// \brief Calculate a 1PNUC.
////
//// Get a 1PNUC out of 10 cold reference frames. The output will depend on the uniformization mode.
//// \param Mat_raw Input streaming matrix
//// \param bufferSizeMax Number of frames to acquire for NUC calculation
void SENSIANuc::calculateNUC1P(cv::Mat *Mat_raw, uint bufferSizeMax)
{
    if(coldMatBuffer.size() >= bufferSizeMax)
        clearBuffers();

    insertReferenceToBuffer(Mat_raw, false);

    if(coldMatBuffer.size() >= bufferSizeMax){
        performNUC1P();
        clearBuffers();
        nucDone = true;
    }
}

//// \brief Get Offset for 1PNUC calculation
////
//// \param coldReference Cold Reference to calculate the offset with (average of frames)
//// \param coldAverage Average of the reference frame (average of averages)
void SENSIANuc::calculateOffset(cv::Mat& coldReference, float coldAverage)
{
    Offset.setTo(coldAverage);
    Offset -= coldReference.mul(Gain);
}

//// \brief Get AdjustMatrix for 1PNUC calculation
////
//// \param coldReference Cold Reference to calculate the offset with (average of frames)
//// \param coldAverage Average of the reference frame (average of averages)
void SENSIANuc::getAdjustMatrixFromReference(cv::Mat& coldReference, float coldAverage)
{
    cv::Mat mat = coldReference.mul(Gain);
    cv::add(mat,Offset,mat);
    cv::divide(coldAverage,mat,mat);
    mat.copyTo(AdjustMatrixU);
}


//// \brief Calculate mean and reference and get offset from them
void SENSIANuc::performNUC1P()
{
    cv::Mat coldReference = getAverageOfBuffer(coldMatBuffer);
    float coldAverage = cv::mean(coldReference).val[0];
    switch (m_uniformizationMode){
        case NORM_USUAL:
            calculateOffset(coldReference,coldAverage);
            break;
        case NORM_PICO:
            getAdjustMatrixFromReference(coldReference,coldAverage);
            break;
    }
}


//// \brief Insert current reference matrix and mean to buffers
//// \param Mat_raw Current frame matrix
//// \param trueMeansHot True if current reference is hot
void SENSIANuc::insertReferenceToBuffer(cv::Mat *mat, bool trueMeansHot)
{
    cv::Mat currentMat;
    mat->convertTo(currentMat,CV_32F);
    if(trueMeansHot)
        hotMatBuffer.push_back(currentMat);
    else
        coldMatBuffer.push_back(currentMat);
}


//// \brief Calculate a 2PNUC.
////
//// Get a 2PNUC out of 10 cold and 10 hot reference frames.
//// \param Mat_raw Input streaming matrix
//// \param bufferSizeMax Number of frames to acquire for NUC calculation
//// \param refCold True if getting/got cold reference
//// \param refHot True if getting/got hot reference
void SENSIANuc::calculateNUC2P(cv::Mat *Mat_raw, uint bufferSizeMax, const bool &refCold, const bool &refHot)
{
    uchar frameCount = getFrameCount(refHot);

    if((!frameCount) && refCold) // Beginning of NUC process
        clearBuffers();

    if( (frameCount < bufferSizeMax) || refHot) // Gathering reference frames to calculate
        insertReferenceToBuffer(Mat_raw, refHot);

    if( (getFrameCount(refHot) == bufferSizeMax) && (refHot) ){ // End of process: calculate output matrices
        calculateGainAndOffset();
        nucDone = true;
        clearBuffers();
    }
}

//// \brief Calculate Gain and Offset matrices out of previous references
void SENSIANuc::calculateGainAndOffset()
{
    cv::Mat coldRef = getAverageOfBuffer(coldMatBuffer);
    cv::Mat hotRef = getAverageOfBuffer(hotMatBuffer);

    float coldAvg = mean(coldRef).val[0];
    float hotAvg = mean(hotRef).val[0];

    calculateGain(coldRef, hotRef, coldAvg, hotAvg);
    calculateOffset(coldRef, coldAvg);
}

/// @brief Check if NUC has been performed
///
/// This function checks if a NUC has been recently performed.
/// @return True if NUC has been recently performed, false otherwise.
bool SENSIANuc::isNucDone() {return nucDone;}


/// @brief End NUC process.
///
/// This function will set the internal NUC state to "finished".
void SENSIANuc::endNuc() {nucDone = false;}


/// \brief Calculate Gain out of cold and hot references.
///
/// This methods calculates Gain matrix out of cold and hot references.
/// The following calculation is performed in order to obtain Gain value:
/// \f[Gain(x,y) =  \frac{\texttt{hotAvg} - \texttt{coldAvg}}{\texttt{hotRef(x,y)} - \texttt{coldRef(x,y)}}\f]
/// for each point (x,y) in the matrix.
/// \param hotRef Hot reference: Average matrix of hot reference frames buffer
/// \param coldRef Cold reference: Average matrix of cold reference frames buffer
/// \param coldAvg Average of frame averages in cold reference
/// \param hotAvg Average of frame averages in hot reference
void SENSIANuc::calculateGain(cv::Mat& coldRef, cv::Mat& hotRef, float coldAvg, float hotAvg)
{
    float* pColdRef = (float*) coldRef.data;
    float* pHotRef = (float*) hotRef.data;
    float* pGain = (float*) Gain.data;

    int maxNumberOfPixels = coldRef.rows * coldRef.cols;
    for(int i = 0; i < maxNumberOfPixels; i++){
        *pGain = (hotAvg - coldAvg) / ((*pHotRef) - (*pColdRef));
        pGain++;
        pHotRef++;
        pColdRef++;
    }
}


//// \brief Get the non-uniformity-corrected matrix out of Gain and Offset matrices
//// \param matOut Output (uniform) matrix
//// \param matIn Input (non-uniform) matrix
void SENSIANuc::getNucMat(cv::Mat *matOut, cv::Mat *matIn)
{ 
    switch (m_uniformizationMode) {
        case NORM_USUAL:
            applyNucUsual(*matIn, *matOut);
            break;
        case NORM_PICO:
            applyNucPico(*matIn, *matOut);
            break;
    }
}


/// @brief Apply usual normalization to matrix
///
/// Usual normlization mode: Output = Input × Gain + Offset
/// @param matIn Input Matrix
/// @param matOut Output Matrix
void SENSIANuc::applyNucUsual(cv::Mat &matIn, cv::Mat &matOut)
{
    if(matIn.empty()) throw std::invalid_argument("NUC Error: Empty input matrix");
    if(Gain.empty()) Gain = cv::Mat(matIn.size(), CV_32F);
    if(Offset.empty()) Offset = cv::Mat(matIn.size(), CV_32F);
    if(matOut.empty()) matOut = cv::Mat(matIn.size(), matIn.type());

    unsigned short *pOut = (unsigned short *) matOut.data;
    unsigned short *pIn = (unsigned short *) matIn.data;

    float *pGain = (float *) Gain.data;
    float *pOffset = (float *) Offset.data;

    int totalNumberOfPixels = matIn.rows * matIn.cols;
    ushort digitalLevel;

    for(int i = 0; i < totalNumberOfPixels; i++) {

        digitalLevel = static_cast<ushort>((*pIn) * (*pGain) + (*pOffset));

        if(digitalLevel>maxDigitalLevelsValue)
            digitalLevel = maxDigitalLevelsValue;

        *pOut = digitalLevel;

        pOut++;
        pIn++;
        pGain++;
        pOffset++;
    }
}


/// \brief Perform element wise matrix multiplication
///
/// \param in1 Input Matrix 1
/// \param in2 Input Matrix 2
/// \param outType Desired OpenCV type of output matrix
cv::Mat SENSIANuc::elementWiseMatMult(cv::Mat& in1, cv::Mat& in2, int outType)
{
    CV_Assert(in1.size() == in2.size());
    int type1 = in1.type(), type2 = in2.type();
    cv::Mat alt = in1.clone();
    cv::Mat* mult = &in2;

    if (type1>type2) {
        in2.convertTo(alt,type1);
        mult = &in1;
    } else if (type1<type2) {
        in1.convertTo(alt,type2);
    }

    alt = alt.mul(*mult);
    alt.convertTo(alt,outType);
    return alt;
}


/// @brief Apply PICO normalization to matrix
///
/// PICO normlization mode: Output = Input × Gain + Offset × AdjustMatrix
/// @param matIn Input Matrix
/// @param matOut Output Matrix
void SENSIANuc::applyNucPico(cv::Mat &matIn, cv::Mat &matOut)
{
    if(matIn.empty()) throw std::invalid_argument("NUC Error: Empty input matrix");
    if(Gain.empty()) Gain = cv::Mat(matIn.size(), CV_32F);
    if(Offset.empty()) Offset = cv::Mat(matIn.size(), CV_32F);
    if(matOut.empty()) matOut = cv::Mat(matIn.size(), matIn.type());

    unsigned short *pOut = (unsigned short *) matOut.data;
    unsigned short *pIn = (unsigned short *) matIn.data;

    float *pGain = (float *) Gain.data;
    float *pOffset = (float *) Offset.data;
    float *pAdjust = (float *) AdjustMatrixU.data;

    int totalNumberOfPixels = matIn.rows * matIn.cols;
    ushort digitalLevel;

    for(int i = 0; i < totalNumberOfPixels; i++) {

        digitalLevel = (*pIn) * (*pGain) + (*pOffset) * (*pAdjust);

        if(digitalLevel>maxDigitalLevelsValue)
            digitalLevel = maxDigitalLevelsValue;

        *pOut = digitalLevel;

        pOut++;
        pIn++;
        pGain++;
        pOffset++;
    }
}


/// @brief Find with regular expression the files within a path
///
/// @param nucPath Path containing .raw files.
/// @param regex Regular expression to find files (e.g., GainU.*\\.raw)
std::vector<std::string> SENSIANuc::findFiles(const std::string& nucPath, const std::regex& regex) {
    std::vector<std::string> foundFiles;
    for (const auto& entry : std::filesystem::directory_iterator(nucPath)) {
        if (std::regex_search(entry.path().filename().string(), regex)) {
            foundFiles.push_back(entry.path().string());
        }
    }
    return foundFiles;
}

/// @brief Load Gain matrix according to specified tint value
///
/// This function will NOT load the nearest NUC files (meaning the ones
/// which are the closest to iValue). It will read the gain file found in Nucs/iValue and, if
/// not found, will re-initialize Gain matrix with initGain().
/// @param iValue Tint value (in us).
void SENSIANuc::setGainForcingNucValue(int iValue)
{
    std::regex regex("GainU.*.raw", std::regex_constants::icase);
    std::string filePath = m_sPathNucStd + "/" + std::to_string(iValue);

    auto rawFiles = findFiles(filePath, regex);

    if (rawFiles.empty())
    {
        initGain();
        throw std::ios_base::failure("GainU.raw not found.");
        return;
    }

    std::string GainPath = rawFiles[0];

    if(!readNucMat(GainPath, Gain, 4))
    {   
        throw std::ios_base::failure("Could not read GainU.raw file.");
        initGain();
    }
}

//// \brief Re-initialize Gain matrix (i.e. set it to 1)
void SENSIANuc::initGain()
{
    mutex.lock();
    Gain.setTo(1.0f);
    mutex.unlock();
}

//// \brief Re-initialize Offset matrix (i.e. set it to 0)
void SENSIANuc::initOffset()
{
    mutex.lock();
    Offset.setTo(0.0f);
    mutex.unlock();
}


/*
/// \brief Gets lower and upper band limits in the specified multi-band offset file
///
/// \param fileOffsetU Multi-band offset file
/// \param down Lower limit
/// \param up Upper limit
void SENSIANuc::readOffsetFileBandsLimits(std::string fileOffsetU, float& down, float& up)
{
    std::ifstream out(fileOffsetU, std::ios::binary);
    if(!out.is_open())
    {
        throw std::ios_base::failure("readOffsetFileBandsLimits: Could not read OffsetU file");
    }

    // ??? 
    // Creo que no hay que tratar out, sino fileOffset, y leerlo de out ?
    out.seekg(0);
    out.read(reinterpret_cast<char*>(&down), sizeof(float));


    // QDataStream out(&fileOffsetU);
    // fileOffsetU.seek(0);
    // out.readRawData(reinterpret_cast<char *>(&down), 4);
    // fileOffsetU.seek(_width*_height*4);
    // out.readRawData(reinterpret_cast<char *>(&up), 4);
}



/// @brief Reads offset file in bands format.
///
/// This function should be called a multi-band offset file is needed.
/// It reads the band limits and gets the current band to be able to read
/// the data as in a usual single-band offset file.
/// @param offsetFilePath Offset file path
/// @param bands Number of bands
/// @param bolometerValue Bolometer value parameter
/// @return False if unable to open the file. True if otherwise.
bool SENSIANuc::readOffsetFileBands(std::string offsetFilePath, uint bands, double bolometerValue)
{
    QString qPath = QString::fromStdString(offsetFilePath);
    QFile fileOffsetU(qPath);
    if (fileOffsetU.open(QIODevice::ReadOnly)){ // Si el fichero se abre...
        float fDownLimit, fUpLimit;
        readOffsetFileBandsLimits(fileOffsetU, fDownLimit, fUpLimit);
        QDataStream out(&fileOffsetU);
        fileOffsetU.seek((_width*_height*4)*(2+getCurrentOffsetBand(bolometerValue, bands, fDownLimit, fUpLimit)));
        mutex.lock();
        out.readRawData(reinterpret_cast<char *>(Offset.data), (_height*_width*4));
        mutex.unlock();
        IFDBG(3){ char sError[200]; snprintf(sError,200, "SENSIANuc::readOffsetFileBands success - %d", qPath.split("/").value(qPath.split("/").length() - 2 ).toInt());  CFileLog::write(sError); }
        fileOffsetU.close();
        return true;
    } else return false;
}
*/

//// \brief Get current band while reading a multi-band offset file.
////
//// \param bolometerValue The value of the bolometer parameter
//// \param bands The number of bands
//// \param fDownLimit Lower band limit
//// \param fUpLimit Upper band limit
uint SENSIANuc::getCurrentOffsetBand(double bolometerValue, int bands,float fDownLimit, float fUpLimit)
{
    //Calculo de la pendiente y ordenada para calcular la banda del archivo OffsetU.raw de cada camara
    double dPendiente = static_cast<double>(bands/static_cast<double>(fUpLimit-fDownLimit));
    double dOrdenada = static_cast<double>(static_cast<double>(bands*fDownLimit)/static_cast<double>(fUpLimit-fDownLimit));

    //Calculamos la banda a leer del archivo.
    int iCurrentBand = static_cast<int>(dPendiente*bolometerValue-dOrdenada);

    //Si la banda calculada está fuera de los extremos, ponemos la banda límite.
    if(iCurrentBand<0) iCurrentBand = 0;
    if(iCurrentBand>bands) iCurrentBand = static_cast<int>(bands-1);

    return bands - iCurrentBand;
}


/// @brief Load PICO sensor Adjust matrix according to specified tint value
///
/// This funciton will NOT load the nearest NUC files (meaning the ones
/// which are the closest to iValue). It will read the gain file found in Nucs/iValue and, if
/// not found, will re-initialize the Adjust matrix with initGain().
/// @param iValue Tint value (in us).
void SENSIANuc::setAdjustMatrixForcingNucValue(int iValue)
{
    std::regex regex("AdjustMatrixU.*.raw", std::regex_constants::icase);
    std::string filePath = m_sPathNucStd + "/" + std::to_string(iValue);

    auto rawFiles = findFiles(filePath, regex);

    if (rawFiles.empty())
    {
        initAdjustMatrix();
        throw std::ios_base::failure("AdjustMatrixU.raw not found.");
        return;
    }

    std::string filePath = rawFiles[0];

    if(!readNucMat(filePath, Gain, 4))
    {   
        throw std::ios_base::failure("Could not read GainU.raw file.");
        initAdjustMatrix();
    }
}

//// \brief Re-initialize PICO sensor Adjust matrix (i.e. set it to 0)
void SENSIANuc::initAdjustMatrix()
{
    mutex.lock();
    AdjustMatrixU.setTo(0.0f);
    mutex.unlock();
}

/// @brief Read Nuc Matrix form disk
/// @param filePath Filename of the file to read the matrix from
/// @param mat Matrix to write the file data to
/// @param size Depth in bytes of the matrix to save (2 for ushort, 4 for float).
/// @return False if unable to read the file. True if otherwise.
bool SENSIANuc::readNucMat(std::string filePath, cv::Mat &mat, int size)
{
    std::ifstream file(filePath, std::ios::binary);

    if(file.is_open())
    {
        std::filesystem::path pathToRaw(filePath);
        int tint = std::stoi(pathToRaw.parent_path().filename().string()); // Guardamos tint de la carpeta (para check)
        int bytesToRead = _height * _width * size; // Bytes a ser leidos
        file.read(reinterpret_cast<char *> (mat.data), bytesToRead); // Leemos
        int bytesRead = file.gcount(); // Bytes leidos

        // Check
        if (bytesRead > 0)
        {
            std::cout << "SENSIANuc::readNucMat. \n Matrix read - " << filePath << "\nWith tint - " << tint << "\nRows/Columns - " << mat.rows << "/" << mat.cols << std::endl;
        }
        else
        {
            std::cout <<"SENSIANuc::readNucMat - no matrix read.\n";
        }

        file.close();
        if (bytesToRead == bytesRead)
        {
            std::cout << "Bytes read = Bytes to read\n";
            return true;
        }
        else
        {
            std::cout << "Bytes read != Bytes to read\nBytes read: " << bytesRead << "\nBytes to read: " << bytesToRead << std::endl; 
            return false;
        }
        
    }
    else
    {
        return false;
    }
}


/// @brief Load Offset matrix according to specified tint value
///
/// This funciton will NOT load the nearest NUC files (meaning the ones
/// which are the closest to iValue). It will read the offset file found in Nucs/iValue and, if
/// not found, will re-initialize the offset matrix with initOffset().
/// @param iValue Tint value (in us).
/// @param bolometerValue Bolometer parameter value.
void SENSIANuc::setOffsetForcingNucValue(double bolometerValue, int iValue)
{
    std::regex regex("OffsetU.*.raw", std::regex_constants::icase);
    std::string filePath = m_sPathNucStd + "/" + std::to_string(iValue);
    auto rawFiles = findFiles(filePath, regex);

    if(rawFiles.empty()){
        initOffset();
        return;
    }
    std::string offsetFile (rawFiles.at(0));
    // uint bands = (uint) offsetFile.size()/(_width*_height*4);
    // std::string filePath {rawFiles.at(0).toStdString()};
    // if(bands > 1){
    //     std::cout << "Implement readOffsetFileBands!\n";
    //     if(!readOffsetFileBands(filePath, bands, bolometerValue)){
    //         initOffset();
    //     }
    // }
    // else if(!readNucMat(offsetFile,Offset,4))
    if(!readNucMat(offsetFile,Offset,4))
    {
        initOffset();
    }
}

/// \brief Calculate PICO sensor Adjust matrix
///
/// \param inputMat input matrix
/// \param bufferSizeMax maximum size of the input matrix buffer
/// \param tint integration time
void SENSIANuc::calculateAdjustMatrix(cv::Mat *inputMat, uint bufferSizeMax, uint tint)
{
    if(getFrameCount()==bufferSizeMax)
        clearBuffers();
    insertReferenceToBuffer(inputMat,false);
    if(getFrameCount()==bufferSizeMax){
        cv::Mat coldReference = getAverageOfBuffer(coldMatBuffer);
        float averageOfColdReference = cv::mean(coldReference).val[0];
        getAdjustMatrixFromReference(coldReference, averageOfColdReference);
        std::string nucDirPath = getDirNuc(m_sPathNucStd, tint);
        saveMat(nucDirPath + "/AdjustMatrixU.raw", AdjustMatrixU, 4);
        clearBuffers();
    }
}