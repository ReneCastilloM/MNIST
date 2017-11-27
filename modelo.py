from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import time

if __name__ == '__main__':
	data = np.loadtxt('mnist_train.csv', delimiter=',') 
	print("Lectura de la base de datos completa")
	ncol = data.shape[1]
	# definiendo entradas y salidas
	X = data[:,1:ncol]
	y = data[:,0]
	tic = time.clock()
	clf = LogisticRegression()
	clf.fit(X, y)
	tac = time.clock()
	print("Entrenamiento completo")
	print("Tiempo de procesador para el entrenamiento (seg):")
	print(tac - tic)
	#Ahora guarda el modelo entrenado
	import pickle
	filename = 'finalized_model.sav'
	pickle.dump(clf, open(filename, 'wb'))