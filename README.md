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
│       ├── images
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

El sistema utiliza dos tipos de datos:

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

### Imágenes de calibración

Ubicación:

```
data/calibration/
```

Cada secuencia contiene imágenes de horizontes acumulados generadas a partir de múltiples frames.


# Ejecución del pipeline

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

Ejecutar:

```
python python/run_video_experiment.py
```


# Resultados

Los resultados se guardan en:

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