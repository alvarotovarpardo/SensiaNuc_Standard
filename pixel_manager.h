#ifndef PIXMAN_H
#define PIXMAN_H

#include <opencv2/opencv.hpp>

/// @brief Estructura para píxel malo.
///
/// Un píxel malo está formado por su posición y por el entorno del que se quiere extraer los valores de mediana para corregirlo.
/// Este entorno dependerá del orden en el que se vaya a aplicar la corrección. Esto asegura robustez al algoritmo y evita que
/// se produzcan situaciones en las que se corrija con valores erróneos. También se traslada una parte importante del procesado (la
/// comprobación de los bordes, por ejemplo) al momento de lectura del archivo de píxeles malos, y simplifica (mucho) la lectura de
/// la función.
struct badPix{
    cv::Point pos;
    std::array<bool,8> neighbourhood;
};


///
/// \brief Clase encargada de administrar los badpix, su deteccion y correccion.
///
/// Al crear un Pixman, le estaremos asignando altura, anchura y un directorio en el que operará.
/// Pixman se encargará entonces de gestionar los píxeles defectuosos de esa cámara, con una matriz cv::Mat
/// de píxeles defectuosos unificada (m_badPixMat). Contiene métodos para leer y escribir los archivos de badpix (readBadPixFile(), saveBadPixFile()),
/// traducirlos de la versión antigua (translateBadPixFileVersion()), construirlos a traves de las matrices de ganancia (getBadPixMatFromGain(), buildBadPixMat()),
/// o aplicarlos para corregir una imagen (applyBadPixCorrection()).
///
/// Las matrices de píxeles defectuosos (bad_pix_v2.raw) tienen la siguiente estructura:
///     \arg Si un píxel no es defectuoso, esa posición tiene asignado el valor 0.
///     \arg Si un píxel es defectuoso y no está en el interior de un cluster, tiene valor 1.
///     \arg Si un píxel es defectuoso y está en el interior de un cluster, tiene un valor asignado acorde con el orden en el que se aplicará la corrección (desde fuera hacia dentro).
///
/// Incluye también métodos para
/// añadir o eliminar píxeles defectuosos a la matriz.
///
class Pixman
{
public:
    Pixman(int height, int width, std::string pathToCamCalibration, uchar bits);

    void addOrDeleteBadPixel(int index, bool trueMeansDelete);
    void analyzeClusters();
    void applyBadPixCorrection(cv::Mat &mat);
    void applyBadPixCorrection(cv::Mat &mat, double min, double max);
    void badPixMatToOrderedList();
    bool buildBadPixMat(std::vector<std::string> gainPaths = {});
    void drawBadPixOverMat(cv::Mat &showMat);
    std::string getBadPixFilename();
    cv::Mat getBadPixMat();
    void getBadPixMatFromGain(cv::Mat &gainMat, cv::Mat &badPixMat , float thmax = 1.3, float thmin = 0.7, float thavg = 0.05, int radio = 2);
    std::vector<std::string> getBadPixPaths(){return m_badPixPaths;}
    std::vector<std::string> getGainPaths(){return m_gainPaths;}
    int getNumberOfBadPix(){return m_badPixList.size();}
    bool isItBad(cv::Point point);
    bool isItBad(int index);
    bool isItBad(int x, int y);
    bool rawToMat(std::string path, cv::Mat &out);
    bool readBadPixFile(std::string badPixFilePath = "");
    void saveBadPixFile(std::string badPixFilePath = "");
    void setBadPixFilename(std::string filename);
    void setBadPixPaths(std::vector<std::string> paths);
    void setFlip(bool bVerticalFlip, bool bHorizontalFlip){m_vFlip = bVerticalFlip; m_hFlip = bHorizontalFlip;}
    void setGainPaths(std::vector<std::string> paths);
    bool translateBadPixFileVersion(std::vector<std::string> nucsSubdirs = {});
    void updateMedianList();

protected:
    int _height,  /*!< Altura de la cámara (filas) */
        _width;  /*!< Anchura de la cámara (columnas) */
    bool m_vFlip, /*!< Fijar a True si hay flip vertical */
         m_hFlip; /*!< Fijar a True si hay flip horizontal */
    uchar m_bits;
    std::string m_sPathCam,  /*!< Ruta a la carpeta de la cámara (01,02,...) */
                m_sBadPixFileName;  /*!< Nombre del raw de pixeles malos (bad_pix_v2.raw) */

    std::vector<std::string> m_badPixPaths,  /*!< Vector con las rutas de los archivos de badpix antiguos, si los hay */
                             m_gainPaths;  /*!< Vector con las rutas de los archivos con valores de gain*/
    cv::Mat m_badPixMat;  /*!< Matriz ordenada con los píxeles defectuosos */
    std::mutex mutex;
    std::vector<badPix> m_badPixList; /*!< Lista con los puntos cuya mediana se aplicará ordenada */
    std::vector<ushort> neighValues; /*!< Lista con los valores en niveles digitales con los que se calcula la mediana */
    std::array<int,8> m_neighToMatPos; /*!< Lista con los miembros del entorno que se usarán para la mediana */
};

#endif // PIXMAN_H
