import numpy
import pickle
from PIL import Image
import time
import linecache

inicial = time.clock()
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
loaded_model_rna = pickle.load(open('finalized_model_RNA.sav', 'rb'))

#print("Tiempo en leer los pickles: " + str(inicial - time.clock()))
#inicial = time.clock()

indice = numpy.random.randint(0,10000)
print ("El registro de NMIST que se eligio fue el " + str(indice + 1))

#print("Tiempo en generar random: " + str(inicial - time.clock()))
#inicial = time.clock()

#data = numpy.loadtxt('mnist.csv', delimiter=',')

file = (linecache.getline('mnist_test.csv', indice))
line = file.split(",")
line[-1] = line[-1].rstrip("\n")
data = []
for i in line:
	data.append(int(i))
real = data[0]
data.pop(0)
vector = [data]
matriz = numpy.asarray(data)

#print("Tiempo en leer archivo: " + str(inicial - time.clock()))
#inicial = time.clock()

predicted = loaded_model.predict(vector)
predicted_rna = loaded_model_rna.predict(vector)

#print("Tiempo en predecir: " + str(inicial - time.clock()))
#inicial = time.clock()

print("Resultado real: " + str(real))
print("Segun el regresor logistico es un: " + str(predicted))
print("Segun la RNA de 3 capas es un: " + str(predicted_rna))

matriz.resize((28,28))

for i in range(28):
	for j in range(28):
		matriz[i][j] = 255 - matriz[i][j]
img = Image.fromarray(matriz)
aumentada = img.resize((200,200))

#print("Tiempo de la imagen: " + str(inicial - time.clock()))

aumentada.show()

print("Tiempo de ejecucion: " + str(time.clock() - inicial))