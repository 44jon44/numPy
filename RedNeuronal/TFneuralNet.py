
import tensorflow._api.v2.compat.v1 as tf
from matplotlib import animation

tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Crea el dataset. Dos circulos concéntricos
# X coordenas-- Y valor del punto
X, Y = make_circles(n_samples=500, factor=0.5, noise=0.05)

# Resolución del mapa de predicción.
res = 100

# Dimensiones del mapa de predicción.
_x0 = np.linspace(-1.5, 1.5, res)
_x1 = np.linspace(-1.5, 1.5, res)

# Input con cada combo de coordenadas del mapa de predicción.
_pX = np.array(np.meshgrid(_x0, _x1)).T.reshape(-1, 2)

# Objeto vacio a 0.5 del mapa de predicción.
_pY = np.zeros((res, res)) + 0.5

# Visualización del mapa de predicción.
plt.figure(figsize=(8, 8))
plt.pcolormesh(_x0, _x1, _pY, cmap="coolwarm", vmin=0, vmax=1)

# Visualización de los colores de la nube de datos.
# plt.scatter(X[:,0],X[:,1],c=Y) sin colores
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c="skyblue")
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c="salmon")

plt.tick_params(labelbottom=False, labelleft=False)

# Definimos los puntos de entrada de la red, para la matriz X e Y.
iX = tf.placeholder('float', shape=[None, X.shape[1]])
iY = tf.placeholder('float', shape=[None])

#lr = 0.01  # learning rate
nn = [2, 16, 8, 1]  # número de neuronas por capa.

# Capa 1 nn[](Posiciones del array nn)
W1 = tf.Variable(tf.random_normal([nn[0], nn[1]]), name='Weights_1')
b1 = tf.Variable(tf.random_normal([nn[1]]), name='bias_1')

# Multiplica datosde entrada por los pesos y añade los sesgos
l1 = tf.nn.relu(tf.add(tf.matmul(iX, W1), b1))

# Capa 2
W2 = tf.Variable(tf.random_normal([nn[1], nn[2]]), name='Weights_2')
b2 = tf.Variable(tf.random_normal([nn[2]]), name='bias_2')

# Multiplica la capa anterior por los pesos y añade los sesgos
l2 = tf.nn.relu(tf.add(tf.matmul(l1, W2), b2))

# Capa 3
W3 = tf.Variable(tf.random_normal([nn[2], nn[3]]), name='Weights_3')
b3 = tf.Variable(tf.random_normal([nn[3]]), name='bias_3')

# Vector de predicciones de Y. Selecciona la primera columna de salida
pY = tf.nn.sigmoid(tf.add(tf.matmul(l2, W3), b3))[:,0]

# Evaluación de las predicciones. Error cuadratico medio
loss = tf.losses.mean_squared_error(pY, iY)

# Definimos al optimizador de la red, para que minimice el error.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

n_steps = 80  # Número de ciclos de entrenamiento.

iPY = []  # Aquí guardaremos la evolución de las predicción, para la animación.

with tf.Session() as sess:
    # Inicializamos todos los parámetros de la red, las matrices W y b.
    sess.run(tf.global_variables_initializer())

    # Iteramos n pases de entrenamiento.
    for step in range(n_steps):

        # Evaluamos al optimizador, a la función de coste y al tensor de salida pY.
        # La evaluación del optimizer producirá el entrenamiento de la red.
        _, _loss, _pY = sess.run([optimizer, loss, pY], feed_dict={iX: X, iY: Y})

        # Cada 25 iteraciones, imprimimos métricas.
        if step % 2 == 0:
            # Cálculo del accuracy.
            acc = np.mean(np.round(_pY) == Y)

            # Impresión de métricas.
            print('Step', step, '/', n_steps, '- Loss = ', _loss, '- Acc =', acc)

            # Obtenemos predicciones para cada punto de nuestro mapa de predicción _pX.
            _pY = sess.run(pY, feed_dict={iX: _pX}).reshape((res, res))

            # Y lo guardamos para visualizar la animación.
            iPY.append(_pY)

# ----- CÓDIGO ANIMACIÓN ----- #

ims = []

fig = plt.figure(figsize=(10, 10))

print("--- Generando animación ---")

for fr in range(len(iPY)):
    im = plt.pcolormesh(_x0, _x1, iPY[fr], cmap="coolwarm", animated=True)

    # Visualización de la nube de datos.
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c="skyblue")
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c="salmon")

    # plt.title("Resultado Clasificación")
    plt.tick_params(labelbottom=False, labelleft=False)

    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
ani.save('animation.mp4')
# ----- CÓDIGO ANIMACIÓN ----- #

ims = []

fig = plt.figure(figsize=(10, 10))

print("--- Generando animación ---")

for fr in range(len(iPY)):
    im = plt.pcolormesh(_x0, _x1, iPY[fr], cmap="coolwarm", animated=True)

    # Visualización de la nube de datos.
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c="skyblue")
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c="salmon")

    # plt.title("Resultado Clasificación")
    plt.tick_params(labelbottom=False, labelleft=False)

    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
ani.save('animation.mp4')
