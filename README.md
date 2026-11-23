# rita-clustering-investigacion
Investigación y desarrollo sobre algoritmos de clustering no supervisados

**1. ¿Qué es el aprendizaje no supervisado y qué es un algoritmo de clustering? ¿En qué se diferencia del aprendizaje supervisado y cuál es su propósito?**

El aprendizaje no supervisado es el tipo de aprendizaje automático que trabaja con datos sin etiquetas. Solo tenemos las características de cada elemento (por ejemplo edad, gasto y visitas de un cliente) y ninguna columna con la respuesta correcta. Lo que le pedimos al algoritmo es que encuentre por sí solo la estructura que hay en esos datos.

Un algoritmo de clustering es el más habitual dentro de este grupo. Su trabajo es agrupar los elementos de modo que los que están en el mismo grupo se parezcan entre sí y se diferencien de los de otros grupos. Lo que agrupa es la similitud global, medida normalmente con una distancia, y no una característica concreta. El resultado son grupos numerados (cluster 0, cluster 1...) y su significado lo interpretamos nosotros mirando qué tienen en común sus miembros.

La diferencia con el aprendizaje supervisado está en las etiquetas. En el supervisado entrenamos al modelo con ejemplos que ya traen la respuesta correcta, como "este cliente compró" o "este no compró". El modelo aprende qué patrones acompañan a cada etiqueta y su propósito es predecir la etiqueta de datos nuevos. Por eso necesita separar los datos en entrenamiento y test: así comprobamos si acierta con casos que no ha visto. En el no supervisado no hay respuesta correcta, así que normalmente no se hace este split y el propósito es explorar y descubrir grupos.

Creo que es importante entender que no supervisado no quiere decir que el modelo decida con total libertad. Nosotros elegimos el algoritmo, la forma de medir la similitud y a menudo el número de grupos. Además, como no hay etiquetas, es más difícil saber si el resultado es bueno. Montarlo es más sencillo que un modelo supervisado, pero evaluarlo e interpretarlo exige más criterio.

Algunas aplicaciones reales del clustering:

- Segmentación de clientes o de mercado, para adaptar campañas a cada grupo.
- Agrupar correos o documentos por temática cuando no tenemos categorías previas. Detectar spam es otro caso, y ese es supervisado, porque se entrena con correos ya marcados.
- Detección de anomalías o fraude, al identificar los elementos que no encajan en ningún grupo.
- Segmentación de imágenes, al agrupar píxeles parecidos.


**2. Explique la diferencia conceptual entre el clustering basado en centroides (particional) y el clustering jerárquico, mencionando un algoritmo representativo de cada enfoque.**




## Bibliografía

- scikit-learn developers. *2.3. Clustering*. scikit-learn documentation. https://scikit-learn.org/stable/modules/clustering.html
