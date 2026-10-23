# 3.5 Aprendizaje no supervisado (Laura)

## Objetivo y representación

El objetivo fue explorar patrones temáticos en las reseñas sin utilizar la etiqueta de sentimiento para
formar los grupos. Se trabajó con una muestra estratificada y reproducible de 8 000 reseñas del conjunto de
entrenamiento. El texto se representó mediante TF-IDF con unigramas y bigramas. TruncatedSVD redujo la matriz
dispersa a 100 componentes y UMAP generó una representación no lineal de 10 dimensiones que prioriza
vecindades locales. Finalmente, HDBSCAN identificó regiones densas y dejó como ruido las observaciones sin
pertenencia clara.

La etiqueta negativo/neutro/positivo no intervino en TF-IDF, SVD, UMAP ni HDBSCAN. Se cruzó con los clusters
después del ajuste y únicamente para interpretarlos.

## Selección de parámetros

Se fijaron `n_neighbors=30`, `min_dist=0`, `min_samples=10` y semilla 42. Se compararon cuatro valores de
`min_cluster_size`:

| min_cluster_size | Clusters | Cobertura | Ruido | Silueta sin ruido |
|---:|---:|---:|---:|---:|
| 50 | 58 | 84,5 % | 15,5 % | 0,674 |
| **100** | **30** | **75,5 %** | **24,5 %** | **0,568** |
| 200 | 13 | 60,5 % | 39,5 % | 0,494 |
| 300 | 8 | 53,7 % | 46,3 % | 0,464 |

Se eligió `min_cluster_size=100` como equilibrio entre cobertura, separación y granularidad. El valor 50
generó 58 clusters, una fragmentación excesiva para interpretar; los valores 200 y 300 aumentaron demasiado
el porcentaje de ruido. La silueta se calculó en el espacio UMAP y, por tanto, no debe compararse de manera
directa con una silueta calculada sobre SVD.

## Resultados e interpretación

Los clusters reflejan patrones de vocabulario y temas, entre ellos: mouse y teclado; cables HDMI/USB;
routers y Wi-Fi; cámaras y lentes; fundas y protectores; audífonos; parlantes; batería y carga; productos que
dejaron de funcionar; problemas de ajuste; baja calidad; y desperdicio de dinero.

El cruce posterior con sentimiento ayuda a interpretar algunos grupos: el cluster asociado con "waste of
money" contiene cerca de 89 % de reseñas negativas; "works great/easy to use" tiene cerca de 79 % positivas;
y "ok/just ok" tiene cerca de 82 % neutras. Estas proporciones no convierten los clusters en clases de
sentimiento ni prueban categorías reales.

## Comparación con otros métodos

- K-Means sobre SVD cubrió todas las reseñas, pero obtuvo siluetas menores a 0,06 y grupos solapados.
- GMM produjo pertenencias probabilísticas, pero su silueta fue 0,0085.
- DBSCAN encontró pequeños núcleos temáticos, pero con 26 clusters dejó 47,9 % como ruido y obtuvo
  silueta negativa; con parámetros más estrictos fragmentó el resultado en 45 clusters y 72,5 % de ruido.
- UMAP–HDBSCAN ofreció la mayor coherencia temática y un mecanismo explícito para tratar casos ambiguos.

## Limitaciones

UMAP modifica la geometría del espacio para conservar relaciones locales y puede acentuar separaciones;
una visualización con "islas" no demuestra por sí sola que existan categorías reales. HDBSCAN es sensible a
`n_neighbors`, `min_cluster_size` y `min_samples`. El 24,5 % marcado como ruido no representa datos erróneos,
sino reseñas sin pertenencia suficientemente clara. Los clusters son patrones exploratorios de lenguaje y
temas, no taxonomías verdaderas de productos ni sustitutos de las etiquetas de sentimiento.
