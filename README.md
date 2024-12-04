### Construcción del contenedor
El directorio de trabajo en ubuntu debe ser el git que se descarga
<br>Luego ejecutar el siguiente comando:<br>
**docker build --no-cache -t zeroshot:latest .**

### Ejecución del contenedor

Para ejecutar el contenedor se debe ejecutar el siguiente comando<br>
**docker run --rm predict:latest NVDA \<pasos a predecir\> \<intervalo de confianza\> \<complejidad puede ser: tiny, mini, small, base, large\>**
El contenedor ejecutara un proceso de predicciones eligiendo entrer 4 tipos de modelos: chronos, granite, timegpt e hibrido.
Se mostraran las predicciones en pantalla.

