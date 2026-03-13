# LensDistortionFromHorizonLines

Este repositorio contiene el código desarrollado para evaluar la estimación y corrección de distorsión radial en imágenes marítimas utilizando líneas de horizonte.

El sistema utiliza imágenes de horizontes acumulados para estimar parámetros de distorsión de lente y posteriormente aplicar la corrección a frames reales de vídeo.
El rendimiento de la corrección se evalúa mediante las métricas PSNR, SSIM y CW-SSIM.

El algoritmo de estimación de distorsión está basado en el método publicado en IPOL y utiliza el código de LensDistortionFromLines.


# Estructura del repositorio

```
LensDistortionFromHorizonLines
│
├── data
│   ├── calibration                     # Imágenes de calibración (horizontes acumulados)
│   │   ├── 0788_calib/
│   │   ├── 0789_calib/
│   │   └── ...
│   │
│   └── raw
│       ├── videos                      # Vídeos originales de entrada
│       ├── images                      # Imágenes de test
│       └── GT_horizon                  # Datos GT de horizontes
│
├── matlab                              # Scripts MATLAB para generar imágenes de horizontes
│   ├── horizontes_GT_binario.m
│   ├── horizontes_GT_submuestreoN.m
│   └── horizontes_GT_submuestreo_frames_uniforme.m
│
├── python                              # Scripts ejecutables principales
│   ├── run_video_experiment.py         # Ejecuta el pipeline completo
│   ├── distorsion.py                   # Aplica distorsión radial sintética
│   ├── desdistorsion_gt.py             # Corrige usando parámetros GT
│   ├── learn_and_apply_undistortion.py # Aprende parámetros y corrige imágenes
│   └── match_opencv_fisheye_direct.py  # Conversión del modelo division a OpenCV
│
├── src/horizon_lens_distortion         # Implementación modular del sistema
│   │
│   ├── distortion                      # Modelos de distorsión
│   │   ├── division_model.py
│   │   └── cropping.py
│   │
│   ├── undistortion                    # Corrección de distorsión
│   │   └── opencv_adapter.py
│   │
│   ├── evaluation                      # Métricas y visualización
│   │   ├── metrics.py
│   │   └── visualization.py
│   │
│   ├── video                           # Procesamiento de vídeo
│   │   ├── io.py
│   │   ├── sampling.py
│   │   └── calibration_cache.py
│   │
│   └── utils
│       └── third_party.py
│
├── third_party
│   └── LensDistortionFromLines         # Código original C++ de estimación de distorsión
│
├── results                             # Resultados generados por los experimentos
│   ├── output_videos
│   ├── plots
│   ├── metrics
│   └── images
│
├── requirements.txt
└── README.md
```


# Requisitos

Sistema recomendado:

```
Linux (Ubuntu 20.04+)
```

Software necesario:

* Python ≥ 3.8
* MATLAB (para generar las imágenes de horizontes acumulados)
* ffmpeg (para extraer frames de vídeo)

Dependencias Python:

```
numpy
opencv-python
scipy
matplotlib
scikit-image
pillow
pywavelets
```

Instalar con:

```
pip install -r requirements.txt
```

También es necesario instalar **ffmpeg** para extraer frames de vídeo:

```
sudo apt install ffmpeg
```


# Compilación del módulo C++

El algoritmo de estimación de distorsión está implementado en **C++** y se utiliza desde Python mediante **pybind11 bindings**.

Para compilarlo:

```
cd third_party/LensDistortionFromLines/python
cmake ..
make
```

Esto generará el módulo Python necesario para ejecutar el pipeline.


# Datos de entrada

Los vídeos y los datos de ground truth utilizados en este proyecto pertenecen al Singapore Maritime Dataset, que contiene secuencias marítimas junto con anotaciones de líneas de horizonte.

El dataset completo puede descargarse en:

https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset

Descarga directa:

https://drive.google.com/file/d/0B43_rYxEgelVb2VFaXB4cE56RW8/view?resourcekey=0-67PrivAOYTGyWxAO_-2n1A

Una vez descargado y descomprimido, los archivos relevantes deben colocarse en las siguientes carpetas del repositorio.

El sistema utiliza **tres tipos de datos de entrada**:

### Vídeos

Ubicación:

```
data/raw/videos/
```

Ejemplo:

```
MVI_0788_VIS_OB.avi
MVI_0789_VIS_OB.avi
...
```

Estos vídeos se utilizan para:

* extraer frames
* aplicar distorsión sintética
* evaluar la corrección de distorsión

---

### Datos GT de líneas de horizonte

Ubicación:

```
data/raw/GT_horizon/
```

Formato:

```
*.mat
```

Estos archivos `.mat` contienen las **líneas de horizonte ground truth** anotadas para cada frame del vídeo.

Se utilizan en los scripts de **MATLAB** para generar las imágenes de horizonte acumulado que se emplean en la estimación de la distorsión.

---

### Imágenes de calibración

Ubicación:

```
data/calibration/
```

Estas imágenes se generan a partir de los **datos GT de horizonte** utilizando los scripts de MATLAB (`horizontes_GT_submuestreo_frames_uniforme.m`), y posteriormente aplicandoles una distorsión a partir de `distorsion.py`



# Ejecución del pipeline

El pipeline completo se compone de varios scripts auxiliares en **Python** y en **MATLAB** para generar las imágenes de horizontes utilizadas en la calibración.

## Generación de imágenes de horizonte (MATLAB)

Antes de ejecutar los experimentos es necesario generar las imágenes de horizontes acumulados a partir de los datos de GT.

Script principal:

```
matlab/horizontes_GT_submuestreo_frames_uniforme.m
```

Este script:

* Genera **imágenes de horizonte acumulado** a partir de los datos de horizonte GT.
* Permite **submuestrear los frames del vídeo** para construir una imagen de calibración más robusta.

Estas imágenes se utilizan posteriormente para **estimar los parámetros de distorsión** (una vez distorsionadas).

---

# Scripts de Python

Los scripts principales se encuentran en:

```
python/
```

### distorsion.py

Aplica **distorsión radial sintética** a una imagen con parámetros conocidos (`d1`, `d2`, `cx`, `cy`).
Se utiliza para generar imágenes distorsionadas de prueba.

---

### desdistorsion_gt.py

Corrige una imagen distorsionada **utilizando los parámetros de distorsión conocidos (ground truth)**.
Permite evaluar la calidad de la corrección cuando los parámetros son correctos. 

---

### learn_and_apply_undistortion.py

Ejecuta el algoritmo de **estimación automática de distorsión** a partir de las imágenes de horizonte distorsionadas y aplica la corrección a la imagen o frame correspondiente.

Internamente llama al módulo C++ mediante los bindings de Python (`lens_distortion_module`). 

---

### match_opencv_fisheye_direct.py

Corrige la distorsión de una imagen formada por rectas paralelas estimando parámetros. 

---

# Ejecución de scripts

Cada script puede ejecutarse de forma independiente desde la raíz del repositorio.

Ejemplo:

```
python python/distorsion.py
```

```
python python/desdistorsion_gt.py
```

```
python python/learn_and_apply_undistortion.py
```

```
python python/match_opencv_fisheye_direct.py
```

---

El script principal es:

```
python/run_video_experiment.py
```

Este script realiza automáticamente:

1. extracción de frames de los vídeos
2. aplicación de distorsión radial sintética
3. estimación de parámetros de distorsión usando horizontes
4. corrección de las imágenes
5. cálculo de métricas de calidad

Para ejecutarlo es necesario primero construir las imágenes de horizontes acumulados a traves de `horizontes_GT_submuestreo_frames_uniforme.m`, aplicarles distorsión con `distorsion.py` y llevarlas a la carpeta de calibración

Ejecutar:

```
python python/run_video_experiment.py
```
Para hacer una prueba mínima de que los códigos funcionan se incluye una imagen dentro de data/raw/images/

# Resultados

Los resultados del script principal se guardan en:

```
results/output_videos/
```

Incluyen:

* imágenes corregidas
* métricas PSNR / SSIM / CW-SSIM
* gráficos por frame
* medias por vídeo
* medias globales


# Referencias

Este repositorio utiliza código basado en:

Hugo Hadfield – **LensDistortionFromLines**
[https://github.com/hugohadfield/LensDistortionFromLines](https://github.com/hugohadfield/LensDistortionFromLines)

AliceVision – **LensDistortionFromLines**
[https://github.com/alicevision/LensDistortionFromLines](https://github.com/alicevision/LensDistortionFromLines)

Algoritmo original publicado en:

Miguel Alemán-Flores, Luis Álvarez, Luis Gómez, Daniel Santana-Cedrés
**Automatic Lens Distortion Correction Using One-Parameter Division Models**
IPOL – Image Processing On Line
[http://www.ipol.im/pub/pre/130/](http://www.ipol.im/pub/pre/130/)


# Autor

Sara Piñón Esteban
Universidad Autónoma de Madrid

[sara.pinnon@estudiante.uam.es](mailto:sara.pinnon@estudiante.uam.es)
[sarapi01@hotmail.com](mailto:sarapi01@hotmail.com)