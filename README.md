# Investigación sobre algoritmos de clustering no supervisados

Autora: Rita Pereira · Bootcamp de Inteligencia Artificial · Noviembre 202

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

La diferencia entre los dos enfoques está en cómo construyen los grupos y en qué tengo que decidir antes de empezar.

El clustering basado en centroides, o particional, divide los datos en un número de grupos que fijo de antemano. Su algoritmo representativo es K-Means. Primero elijo K, que es el número de grupos. El algoritmo coloca K centroides al azar y asigna cada punto al centroide más cercano. Después mueve cada centroide al centro de los puntos que le han tocado y repite el proceso hasta que dejan de moverse. El resultado es una única división de los datos en K grupos, y como el punto de partida es aleatorio, dos ejecuciones pueden dar resultados algo distintos.

El clustering jerárquico no necesita que fije el número de grupos al principio. Su algoritmo representativo es el jerárquico aglomerativo. Empieza con cada punto como su propio grupo y va uniendo los dos grupos más cercanos hasta que todo queda en uno solo. El resultado es un dendrograma, un árbol con todas las uniones, y obtengo los grupos cortándolo a la altura que quiera. Aquí sí tengo que decidir cómo mido la distancia entre grupos, lo que se llama criterio de enlace o linkage. Ward, que une los grupos cuya fusión aumenta menos la variación interna, es el más usado y el que scikit-learn aplica por defecto.

Las dos formas se llevan bien con casos distintos. Según la guía de scikit-learn, K-Means escala a muestras muy grandes y a un número medio de clusters, y es de uso general con grupos de tamaño parecido y geometría plana. El aglomerativo maneja muchos clusters, admite restricciones de conectividad y distancias no euclídeas, y es más lento cuando crece el número de datos. Otra diferencia es que K-Means es inductivo, así que puede asignar un punto nuevo a un grupo sin repetir el proceso, y el aglomerativo es transductivo, así que si llegan datos nuevos hay que volver a calcularlo.

Un ejemplo de por qué importa el enfoque: con dos medias lunas entrelazadas, K-Means falla porque asigna cada punto al centro más cercano y corta las lunas por la mitad. El aglomerativo con linkage single sí puede separarlas, porque une los puntos más cercanos entre sí y sigue la curva de cada luna.

Resumiendo, K-Means pide K desde el inicio y devuelve una sola división, y el jerárquico no lo pide y devuelve un árbol con todos los niveles de agrupación.

Fuente: scikit-learn developers, 2.3. Clustering (consultado el 23 de noviembre de 2026).

**3. ¿En qué consiste el clustering basado en densidad (como DBSCAN) y cómo difiere su enfoque del clustering basado en centroides y del jerárquico?** 




## Bibliografía

- scikit-learn developers. *2.3. Clustering*. scikit-learn documentation. https://scikit-learn.org/stable/modules/clustering.html
