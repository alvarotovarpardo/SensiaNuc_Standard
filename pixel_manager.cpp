#include "pixel_manager.h"
#include <fstream>
#include <codecvt>




/*  TODO:
 *
 *  -> Ver como hacer para salvar la duplicidad con los paths de las matrices de gain (al implementar).
 *  -> Disimular la correccion de clusters grandes con una desviacion de la mediana.
 *  -> Mejorar la deteccion de BP en los casos de cluster.
 *
*/










///
/// \brief Constructor de la clase Pixman.
///
///
/// \param height altura de la imagen (rows)
/// \param width anchura de la imagen (cols)
/// \param pathToCamCalibration ruta a la carpeta de la camara (01, 02, ...)
///
Pixman::Pixman(int height, int width, std::string pathToCamCalibration, uchar bits)
{
    _height = height;
    _width = width;
    m_sPathCam = pathToCamCalibration;
    m_badPixMat = cv::Mat::zeros(height,width,CV_16UC1);
    m_neighToMatPos={-_width-1,-_width,-_width+1,-1,1,_width-1,_width,_width+1};
    m_sBadPixFileName = "/bad_pix_v2.raw";
    if(bits!=0) m_bits = bits; else m_bits = 16;
    m_vFlip = false;
    m_hFlip = false;
}











/*  Índice:
 *
 *      addOrDeleteBadPixel
 *      analyzeClusters
 *      applyBadPixCorrection
 *      badPixMatToOrderedList
 *      buildBadPixMat
 *      drawBadPixOverMat
 *      getBadPixFilename
 *      getBadPixMat;
 *      getBadPixMatFromGain
 *      getBadPixPaths
 *      getGainPaths
 *      isItBad
 *      rawToMat
 *      readBadPixFile
 *      saveBadPixFile
 *      setBadPixFilename
 *      setBadPixPaths
 *      setGainFilename
 *      setGainPaths
 *      translateBadPixFileVersion
 *      updateMedianList
 *
 */



///////
/// \brief Añade o elimina un badpix de la matriz.
///
/// Esta función añade un píxel defectuoso a la matriz *m_badPixMat* o lo elimina, en función del valor de *trueMeansDelete*
/// (como su propio nombre indica, true implica eliminar el badpix de la matriz). El píxel queda especificado por *index*,
/// el índice del píxel dentro de la matriz dado por el orden lexicográfico en x e y.
///
/// **Ejemplo**: En una matriz 640x512 (640 filas, 512 columnas) el elemento dado por (0,4) tiene índice 4.
/// El elemento dado por (1,4) tiene índice 516. *addOrDeleteBadPixel(516,false)* añadirá un píxel defectuoso a la *m_badPixMat*
/// en la posición (1,4).
///
/// \param index Índice del píxel
/// \param trueMeansDelete Eliminar o añadir píxel
///
void Pixman::addOrDeleteBadPixel(int index, bool trueMeansDelete){
    ushort *badPixArray = (ushort *) m_badPixMat.data;
    if((index >= _height*_width)||(index < 0)){
        throw std::out_of_range("addOrDeleteBadPixel(): Pixel index out of range");
        return;
    }
    if(m_vFlip) index = (_height-floor(index/_width)-1)*_width + index%_width;
    if(m_hFlip) index = _width*floor(index/_width) + _width - index%_width - 1;
    mutex.lock();
    if(trueMeansDelete){
        badPixArray[index] = 0;
    }else{
        badPixArray[index] = 1;
    }
    mutex.unlock();
    updateMedianList();
}










///
/// \brief Toma la matriz miembro de badpixels y la convierte en matriz con puntos ordenados
///
/// Esta función toma *m_badPixMat* y le aplica el siguiente algoritmo:
/// -# Lleva todos los defectuosos a 1, y los que no son defectuosos se quedan en 0.
/// -# Convoluciona la matriz con un kernel \f$  3 \times 3  \f$ para obtener un matriz (\f$M_3\f$) con la suma del entorno \f$3 \times 3\f$ de ese punto en la matriz original.
/// Hace lo mismo con un kernel \f$5 \times 5\f$ para dar \f$M_5\f$.
/// -# Se toman sólo los elementos con 5 o más adyacentes (su valor en \f$M_3\f$ es >5)
/// -# Ordena estos puntos lexicográficamente en su valor en \f$M_3\f$ y en su valor en \f$M_5\f$. (Ver ejemplo)
///
/// El resultado es una matriz con 0 en los píxeles no defectuosos, 1 en los píxeles defectuosos con menos de 4 defectuosos adyacentes,
/// y valor *n* con *n>1* para los píxeles defectuosos con 4 o más defectuosos adyacentes, ordenados de menor a mayor número de adyacentes en \f$M_3\f$
/// y después de menos número de adyacentes en \f$M_5\f$ a más. Los píxeles que tengan el mismo par de números de adyacentes tendrán también el mismo valor asignado.
///
/// **Ejemplo**: Sean píxeles \f$p_1, p_2, p_3, p_4\f$ tales que:
/// \arg \f$M_3(p_1) = 4\f$, \f$M_5(p_1) = 12 \f$
/// \arg \f$M_3(p_2) = 7\f$, \f$M_5(p_2) = 10 \f$
/// \arg \f$M_3(p_3) = 7\f$, \f$M_5(p_3) = 7 \f$
/// \arg \f$M_3(p_4) = 8\f$, \f$M_5(p_4) = 8 \f$
///
///
/// donde \f$M(p)\f$ representa el valor de la matriz \f$M\f$ en el punto \f$p\f$. Entonces se asigna valor 1 a \f$p_1\f$, y valores
/// 2,3,4 a \f$p_3, p_2, p_4\f$ respectivamente.
///
///
void Pixman::analyzeClusters(){

    //Obtenemos los píxeles malos no aislados de la matriz (los que tienen valor >1)
    std::vector<cv::Point> nonZeroVector;
    cv::findNonZero(m_badPixMat>1,nonZeroVector);

    //Inicializamos matrices
    cv::Mat kernel3 = cv::Mat::ones(3,3,CV_32F), kernel5 = cv::Mat::ones(5,5,CV_32F),
            clustercenters_mat = cv::Mat::zeros(_height,_width,CV_8UC1), clustercenters_masked = cv::Mat::zeros(_height,_width,CV_8UC1),
            buscaminas = cv::Mat::zeros(_height,_width,CV_8UC1), buscaminas2 = cv::Mat::zeros(_height,_width,CV_8UC1);

    //Nos hacemos una copia "local" de la matriz general de BP
    mutex.lock();
    cv::Mat badpixel_mat = m_badPixMat.clone();
    mutex.unlock();

    //Si hay alguno no aislado (>1), lo llevamos a 1 para repetir el análisis
    if(!nonZeroVector.empty()){
        cv::threshold(badpixel_mat,clustercenters_mat,0,1,cv::THRESH_BINARY);
        clustercenters_mat.convertTo(clustercenters_mat,CV_8UC1);
        badpixel_mat.setTo(1,clustercenters_mat);
    }

    //La convertimos en CV_8UC1
    badpixel_mat.convertTo(badpixel_mat,CV_8UC1);

    //Calculamos la suma del bloque de 3 que rodea cada punto.
    cv::filter2D(badpixel_mat,clustercenters_mat,CV_8UC1,kernel3,cv::Point(-1,-1),0,cv::BORDER_REPLICATE);

    //Suma en bloques de 5. Nos interesa para clusters grandes (se admiten hasta 30 px/cluster!)
    cv::filter2D(badpixel_mat,buscaminas,CV_8UC1,kernel5,cv::Point(-1,-1),0,cv::BORDER_REPLICATE);

    //Tomamos solamente aquellos que son "centros" de cluster (y nos aseguramos de que son malos)
    clustercenters_mat.copyTo(clustercenters_masked,badpixel_mat);

    //Convertimos la matriz local de vuelta al formato de la general
    badpixel_mat.convertTo(badpixel_mat,CV_16UC1);

    //Metemos los centros anteriores en orden creciente de numero de adyacentes
    double minval,maxval;
    ushort count=2;
    for(auto k3=5;k3<10;k3++){
        buscaminas2.setTo(0);
        buscaminas.copyTo(buscaminas2,clustercenters_masked==k3);
        cv::minMaxLoc(buscaminas2,&minval,&maxval);
        for(auto k5=k3;k5<=maxval;k5++){
            cv::findNonZero(buscaminas2==k5,nonZeroVector);
            for(auto p:nonZeroVector){
                badpixel_mat.at<ushort>(p) = count;
            }
            count++;
        }
    }

    //Nos llevamos el cálculo de la matriz local a la global
    mutex.lock();
    badpixel_mat.copyTo(m_badPixMat);
    mutex.unlock();
}










///
/// \brief Aplica la mediana a la matriz mat en los puntos dados por el archivo de badpix cargado
///
/// En primer lugar aplica la corrección (median blur) a los píxeles defectuosos simples (píxeles con menos de 5 defectuosos adyacentes).
/// Después aplica la corrección a los píxeles internos (5 o más defectuosos adyacentes) ordenados de dentro hacia afuera, según el orden
/// dado por analyzeClusters(). Para esto último es necesario haber ejecutado previamente badPixelToOrderedList().
///
/// \param mat Matriz sobre la que aplicar la corrección.
///
void Pixman::applyBadPixCorrection(cv::Mat &mat){
    if(m_badPixList.empty()) badPixMatToOrderedList();
    unsigned short *pMat = (unsigned short *) mat.data;

    neighValues={};
    for(auto p:m_badPixList){
        for(auto k=0;k<8;k++){
            if(p.neighbourhood[k])
                neighValues.push_back(*(pMat+(p.pos.y)*_width + p.pos.x+m_neighToMatPos[k]));
        }
        if(neighValues.size()>0)
        {
            std::nth_element(neighValues.begin(), neighValues.begin()+neighValues.size()/2,neighValues.end()); //Obtenemos la mediana
            *(pMat+(p.pos.y)*_width + p.pos.x) = neighValues[neighValues.size()/2]; //Actualizamos el valor de la matriz a la mediana
        }
        neighValues.clear();
    }
}



///
/// \brief Funcion equivalente a la anterior, salvo porque introduce una desviacion a la mediana para disimular la correccion en
/// clusters grandes. La desviacion se hara con un numero pseudoaleatorio que es fijo para cada pixel, de forma que actue como algo parecido
/// al offset de la imagen.
/// \param mat Matriz a corregir
/// \param min Minimo del rango dinamico de la imagen
/// \param max Maximo del rango dinamico de la imagen
///
void Pixman::applyBadPixCorrection(cv::Mat &mat, double min, double max){
    if(m_badPixList.empty()) badPixMatToOrderedList();
    float fact = 0.05; //Variacion dentro del 5% del rango dinamico
    unsigned short *pMat = (unsigned short *) mat.data;

    neighValues={};
    for(auto p:m_badPixList){
        for(auto k=0;k<8;k++){
            if(p.neighbourhood[k])
                neighValues.push_back(*(pMat+(p.pos.y)*_width + p.pos.x+m_neighToMatPos[k]));
        }
        int rval = ((int)round(1.5*(p.pos.y)*pow(p.pos.x,1.5)))%((int)pow(2,m_bits)-1) - pow(2,m_bits-1); //Ajustamos al maximo de ND
        rval=round(fact*rval*(max-min)/pow(2,m_bits-1)); // Transformamos al valor relativo en rango dinamico
        if(neighValues.size()>0)
        {
            std::nth_element(neighValues.begin(), neighValues.begin()+neighValues.size()/2,neighValues.end()); //Obtenemos la mediana
            *(pMat+(p.pos.y)*_width + p.pos.x) = neighValues[neighValues.size()/2] + (ushort) rval; //Actualizamos el valor de la matriz a la mediana
        }
    }
}











///
/// \brief Construye la lista ordenada de puntos para mediana a partir del badpix
///
/// Toma la matriz de píxeles defectuosos (en su v2, esto es, con valores ordenados) y construye una lista con los píxeles internos ordenados
/// desde fuera hacia dentro, con el orden dado por analizeClusters().
///
void Pixman::badPixMatToOrderedList(){
    double minval,maxval;
    std::vector<cv::Point> nonzerovector;
    std::vector<badPix> badPixListAux;
    std::array<bool,8> neigh;
    std::vector<std::array<int,2>> relatCoord = {{-1,-1},{0,-1},{1,-1},{-1,0},{1,0},{-1,1},{0,1},{1,1}};


    mutex.lock();
    cv::Mat badMat = m_badPixMat.clone();
    mutex.unlock();

    cv::minMaxLoc(badMat,&minval,&maxval);

    for(int k=1;k<=maxval;k++){
        cv::findNonZero(badMat==k,nonzerovector);
        for(auto p:nonzerovector){
            for(auto j=0;j<8;j++){
                if( // Si el que estamos mirando verifca que:
                    !(p.x==0 && (j==0 || j==3 || j==5)) && // No esta fuera por la izq,
                    !(p.x==_width-1 && (j==2 || j==4 || j==7)) && // No esta fuera por la dcha,
                    !(p.y==0 && (j<3)) && // No esta fuera por arriba,
                    !(p.y==_height-1 && (j>4)) && // No esta fuera por abajo
                    (badMat.at<ushort>(cv::Point(p.x+relatCoord[j][0],p.y+relatCoord[j][1])) == 0) // Y es bueno,
                    )
                    neigh[j] = true; // Entonces lo incoroporamos a la mediana
            }
            badPixListAux.push_back({p,neigh});
            badMat.at<ushort>(p)= (ushort) 0; // Como ya lo habríamos corregido, lo llevamos a cero para que se meta en el siguiente.
            neigh = {false,false,false,false,false,false,false,false};
        }
    }

    mutex.lock();
    m_badPixList.clear();
    m_badPixList.insert(m_badPixList.end(), badPixListAux.begin(), badPixListAux.end());
    mutex.unlock();
}










////
/// \brief Construye la matriz de BadPix a partir de todas las matrices de Gain alojadas en gainPaths.
///
/// Esta función une las matrices de píxeles defectuosos obtenidas a partir de las matrices de Gain en m_gainPaths. Se tiene entonces
/// una única matriz de píxeles defectuosos, construida con la información de las uniformizaciones a todos los tiempos de integración.
///
/// \param gainPaths Vector de strings con los paths a las matrices de gain de las que obtener los píxeles defectuosos.
///
bool Pixman::buildBadPixMat(std::vector<std::string> gainPaths){
    if(gainPaths.empty()) gainPaths = m_gainPaths;
    bool foundGainFiles=false;
    cv::Mat gainMat(_height,_width,CV_32FC1), tempBadPix(_height,_width,CV_16UC1),
            orBadPix = cv::Mat::zeros(_height,_width,CV_16UC1);
    for(auto s:gainPaths){
        if(rawToMat(s,gainMat)){
            foundGainFiles=true;
            getBadPixMatFromGain(gainMat,tempBadPix);
            cv::bitwise_or(tempBadPix,orBadPix,orBadPix);
        }
    }
    if(foundGainFiles){
        mutex.lock();
        orBadPix.copyTo(m_badPixMat);
        mutex.unlock();
        return true;
    }else{
        return false;
    }
}























///
/// \brief Pinta los badpix encima de la matriz showMat
///
/// Esta función pinta los píxeles defectuosos (*m_badPixMat*) en rojo, encima de la matriz *showMat*.
///
/// \param showMat Matriz sobre la que pintar los píxeles malos
///
void Pixman::drawBadPixOverMat(cv::Mat &showMat){
    cv::Mat badPixMask;
    m_badPixMat.convertTo(badPixMask,CV_8UC1);
    if (m_vFlip) cv::flip(badPixMask, badPixMask, 0);
    if (m_hFlip) cv::flip(badPixMask, badPixMask, 1);
    cv::Mat matRed = cv::Mat(showMat.rows,showMat.cols,showMat.type(),cv::Scalar(0,0,255));
    matRed.copyTo(showMat,badPixMask);
}




















///
/// \brief Obtiene los badpix a partir de la matriz de Gain, usando un criterio de adyacencia.
///
/// Esta función crea una matriz de píxeles defectuosos no ordenada, con valor "1" para aquellos píxeles que sean defectuosos y con un "0"
/// para aquellos que no lo sean. Para hacerlo, se aplican dos criterios:
///
/// \arg Umbral sobre la ganancia. Por defecto se considerarán píxeles no defectuosos aquellos con ganancia situada entre 0.7 y 1.3.
/// \arg Umbral de adyacencia. Se impone un valor máximo para la diferencia que un píxel tiene (en ganancia) con la media de sus adyacentes.
/// Viene definido por *thavg*.
///
/// \param gainMat Input: Matriz de Gain
/// \param badPixMat Output: Matriz de BadPix
/// \param thmax Umbral superior para el gain (si gain>thmax, malo) -- default = 1.3
/// \param thmin Umbral inferior para el gain (si gain<thmin, malo) -- default = 0.7
/// \param thavg Maxima variacion con respecto al entorno para considerarlo malo -- default = 0.08
/// \param radio Radio del entorno a evaluar para el criterior de adyacencia -- default = 2
///
void Pixman::getBadPixMatFromGain(cv::Mat &gainMat, cv::Mat &badPixMat, float thmax, float thmin, float thavg, int radio){
    cv::Mat correctedGainMat = gainMat.clone();
    cv::Mat matgainaux = cv::Mat::zeros(_height,_width, CV_32F), matones32 = cv::Mat::ones(_height,_width,CV_32F),
            matbadpix = cv::Mat::zeros(_height,_width,CV_16UC1), matnewval,
            kernel_radio = cv::Mat::ones(radio*2+1,radio*2+1,CV_32F), matones16 = cv::Mat::ones(_height,_width,CV_16UC1);

    cv::medianBlur(correctedGainMat,matnewval,5);

    cv::bitwise_or(correctedGainMat>thmax,correctedGainMat<thmin,matgainaux);
    matgainaux.convertTo(matgainaux,CV_8UC1);
    matones16.copyTo(matbadpix,matgainaux); //Nos quitamos los cantosos de un plumazo

    //Llevamos los que sabemos que son malos a un valor estándar (la mediana, en este caso) para poder distinguir el resto
    matgainaux = (correctedGainMat>thmax); matgainaux.convertTo(matgainaux,CV_8UC1);
    matnewval.copyTo(correctedGainMat,matgainaux);
    matgainaux = (correctedGainMat<thmin); matgainaux.convertTo(matgainaux,CV_8UC1);
    matnewval.copyTo(correctedGainMat,matgainaux);

    //Nos quitamos los NaN
    matgainaux = (correctedGainMat != correctedGainMat);
    matgainaux.convertTo(matgainaux,CV_8UC1);
    matnewval.copyTo(correctedGainMat,matgainaux);

    unsigned short* pmatbadpix = (unsigned short*) matbadpix.data;
    float mu;

    cv::filter2D(correctedGainMat,matgainaux,CV_32F,kernel_radio,cv::Point(-1,-1),0,cv::BORDER_REPLICATE);

    for(auto i=0; i<_height; i++){
        for(auto j=0; j<_width; j++){
            mu = ((matgainaux.ptr<float>(i)[j])-(correctedGainMat.ptr<float>(i)[j]))/((radio*2+1)*(radio*2+1)-1); //La media del entorno
            if(abs(mu-correctedGainMat.ptr<float>(i)[j])>thavg){
                correctedGainMat.ptr<float>(i)[j] = mu;
                *pmatbadpix = (ushort) 1;
            }
            if(i*j<_height*_width)
                pmatbadpix++;
        }
    }
    mutex.lock();
    matbadpix.copyTo(badPixMat);
    mutex.unlock();
}





///
/// \brief Devuelve el nombre del archivo badpix. Formato: "/filename.raw"
///
/// Por defecto: "/bad_pix_v2.raw".
///
/// \param filename nombre del archivo
///
std::string Pixman::getBadPixFilename(){
    return m_sBadPixFileName;
}






//////
/// \brief Devuelve la badPixMat
/// \return badPixMat Matriz de pixeles defectuosos
///
cv::Mat Pixman::getBadPixMat(){
    cv::Mat badPixMat = m_badPixMat.clone();
    if (m_vFlip) cv::flip(badPixMat, badPixMat, 0);
    if (m_hFlip) cv::flip(badPixMat, badPixMat, 1);
    return badPixMat;
}



///
/// \brief Devuelve verdadero si el píxel de la posición fijada es defectuoso
/// \param point Posición del píxel
/// \return
///
bool Pixman::isItBad(cv::Point point){
    if((point.x<0) || (point.x >= _width) || (point.y<0) || (point.y >= _height)){
        throw std::out_of_range("isItBad(): Pixel index out of range");
        return false;
    }
    if(m_badPixMat.at<ushort>(point))
        return true;
    else
        return false;
}

///
/// \brief Devuelve verdadero si el píxel de la posición fijada es defectuoso
/// \param index Posición del píxel
/// \return
///
bool Pixman::isItBad(int index){
    ushort *badPixArray = (ushort *) m_badPixMat.data;
    if((index<0)||(index >= _height*_width)){
        throw std::out_of_range("isItBad(): Pixel index out of range");
        return false;
    }
    if(badPixArray[index])
        return true;
    else
        return false;
}

///
/// \brief Devuelve verdadero si el píxel de la posición fijada es defectuoso
/// \param x Posición x del píxel
/// \param y Posición y del píxel
/// \return
///
bool Pixman::isItBad(int x, int y){
    if((x>=_width) || (y>=_height) || (x<0) || (y<0)){
        throw std::out_of_range("isItBad(): Pixel index out of range");
        return false;
    }
    if(m_badPixMat.at<ushort>(y,x))
        return true;
    else
        return false;
}

////
/// \brief Escribe el RAW de path en out, una Mat de tipo CV_16UC1 o CV_32FC1
///
/// \param path Path al archivo .raw a leer
/// \param out Matriz CV_16UC1 o CV_32FC1
/// \return Verdadero si se lee con exito y el formato es correcto, falso en caso contrario
///
bool Pixman::rawToMat(std::string path, cv::Mat &out){
    #ifndef UNIX_SO
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> UTF8toUTF16;
    std::wstring wPath = UTF8toUTF16.from_bytes(path);
    std::ifstream file(wPath, std::ios::binary);
    #else
    std::ifstream file(path, std::ios::binary);
    #endif

    if(file.fail())
        return false;
    if(out.type() == CV_16UC1){
        char buffer[sizeof(ushort)];
        ushort *array = (ushort *) out.data;
        int i = 0;
        mutex.lock();
        while (file.read(buffer, sizeof(buffer))&&(i<out.rows*out.cols)){
            memcpy(&array[i], buffer, sizeof(buffer));
            i++;
        }
        file.close();
        mutex.unlock();
        return true;
    }else if(out.type() == CV_32FC1){
        char buffer[sizeof(float)];
        float *array = (float *) out.data;
        int i = 0;
        mutex.lock();
        while (file.read(buffer, sizeof(buffer))&&(i<out.rows*out.cols)){
            memcpy(&array[i], buffer, sizeof(buffer));
            i++;
        }
        file.close();
        mutex.unlock();
        return true;
    }else{
        return false;
    }
}
























///
/// \brief Lee el archivo badpix del path por defecto
///
/// Esta función intenta leer el archivo badpix nuevo en la ruta especificada o, por defecto, en el directorio dado al constructor
/// de Pixman. Si lee el archivo, realiza el proceso de análisis de clusters de píxeles (ver analyzeClusters()) y lo guarda.
/// Si no encuentra este archivo, entonces intenta traducir los archivos bad_pix.raw antiguos
/// (especificados previamente con setBadPixPaths()).
///
/// Nota: La traducción se deja de momento fuera de esta función.
///
/// \return True si ha leido bad_pix_v2 o si ha traducido correctamente un bad_pix. False si no habia ninguno de los anteriores.
///
bool Pixman::readBadPixFile(std::string badPixFilePath){
    if(badPixFilePath.empty()) badPixFilePath = m_sPathCam+m_sBadPixFileName;
    if(!rawToMat(badPixFilePath,m_badPixMat)){
        return false; //return translateBadPixFileVersion(m_badPixPaths);
    }else{
        return true;
    }
}


















////
/// \brief Guarda el archivo de badpix ordenado en el path raiz del Pixman.
/// \param badPixFilePath Ruta del archivo, si se quiere especificar. Por defecto es el path raiz.
///
void Pixman::saveBadPixFile(std::string badPixFilePath){
    if(badPixFilePath.empty()) badPixFilePath = m_sPathCam + m_sBadPixFileName;
    #ifndef UNIX_SO
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> UTF8toUTF16; //SOLO VALE PARA WINDOWS
    std::wstring bpFPwstring = UTF8toUTF16.from_bytes(badPixFilePath);
    std::ofstream badPixFile(bpFPwstring, std::ios::binary);
    #else
    std::ofstream badPixFile(badPixFilePath, std::ios::binary);
    #endif
    if(badPixFile.is_open()){
        char buffer[sizeof(ushort)];
        ushort *badPixArray = (ushort *) m_badPixMat.data;
        mutex.lock();
        for(int i=0; i<_height*_width; i++){
            memcpy(buffer, &badPixArray[i], sizeof(ushort));
            badPixFile.write(buffer,sizeof(buffer));
        }
        badPixFile.close();
        mutex.unlock();
    }
}







///
/// \brief Fijar nombre de archivo para el badpix ordenado. Formato: "/archivo.raw"
///
/// \param filename nuevo nombre de archivo
///
void Pixman::setBadPixFilename(std::string filename){
    m_sBadPixFileName = filename;
}







/////
/// \brief Guarda los paths de las matrices de badpix
/// \param Vector con los paths de las matrices de badpix
///
void Pixman::setBadPixPaths(std::vector<std::string> paths){
    m_badPixPaths = paths;
}












/////
/// \brief Guarda los paths de las matrices de gain
/// \param Vector con los paths de las matrices de gain
///
void Pixman::setGainPaths(std::vector<std::string> paths){
    m_gainPaths = paths;
}















////
/// \brief Transforma badpix antiguos en el nuevo y lo guarda
///
/// Esta función traduce los archivos *bad_pix.raw* (cuyas rutas están especificadas en *oldBadPixRaws*), uniéndolos y aplicándoles
/// análisis de clusters (ver analyzeClusters()). Guarda el archivo nuevo en el path raíz de Pixman.
///
/// \param oldBadPixRaws Vector que contiene los paths de los badpix antiguos
///
bool Pixman::translateBadPixFileVersion(std::vector<std::string> oldBadPixRaws){
    cv::Mat newBadPix = cv::Mat::ones(_height,_width,CV_16UC1);
    cv::Mat oldBadPix = newBadPix.clone();
    bool oldFilesFound=false;
    for(auto s:oldBadPixRaws){
        if(rawToMat(s,oldBadPix)){
            oldFilesFound=true;
            cv::bitwise_and(oldBadPix,newBadPix,newBadPix);
        }
    }
    if(oldFilesFound){
        cv::bitwise_not(newBadPix,newBadPix);
        newBadPix = newBadPix-65534; //Provisional
        mutex.lock();
        newBadPix.copyTo(m_badPixMat);
        mutex.unlock();
        analyzeClusters();
        saveBadPixFile();
        badPixMatToOrderedList();
        return true;
    }else{
        return false;
    }
}

////
/// \brief Actualiza el vector de badpix (WILL DEPRECATE)
///
/// Esta función guarda en un vector los puntos con valor 1 de la matriz de píxeles malos (esto es, los píxeles malos simples).
///
void Pixman::updateMedianList(){
//    mutex.lock();
//    cv::findNonZero(m_badPixMat==1,median_list);
//    mutex.unlock();
}
