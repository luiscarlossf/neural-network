"""
Rede Neural que seja capaz de classificar se determnada
pessoa joga futebol ou tênis. As entradas devem ocorrer na 
sequência: Nadal(11), Gabriel(01), Federer(10), Nadal(00) 
e novamente Nadal(11) para verificar se a rede aprendeu.
"""
import math
import numpy as np
from random import uniform

class Perceptron:
    def __init__(self):
        self.weights = None

    def active_function(self, inputs, threshold):
    	result = ((inputs * self.weights).sum())
    	print("\n------------------------------------")
    	print("Activation Function(input * weights) = ")
    	print("input: ", inputs)
    	print("weights: ", self.weights)
    	print("threshold: ", threshold)
    	print("input * weights: ", inputs * self.weights)
    	print("result: ", result)
    	print("---------------------------------------\n")
    	if math.tanh(result) > threshold:
    		return 1
    	else:
    		return 0
    	#return int(math.tanh(result))

class NeuralNetwork:
	def __init__(self, size):
		self.perceptrons = [Perceptron() for i in range(0, size)]

	def training(self, inputs, outputs, learning_ratio, threshold):
		inputs = np.array(inputs)
		outputs = np.array(outputs)
		weights = None
		for perceptron in self.perceptrons:
			weights = np.array([0 for i in range(0, inputs.shape[1])])
			perceptron.weights = weights
		epoca = 0
		print("===============TRAINING==================")
		print("Input: ", inputs)
		print("Desire Output: ", outputs)
		print("Learning Ratio: ", learning_ratio)
		print("Weights: ", weights)
		print("=========================================")

		while True:
			#errors = np.array([False] * inputs.shape[0])
			errors = True
			for index, i in enumerate(inputs):
				for perceptron  in self.perceptrons:
					output = perceptron.active_function(i, threshold)
					if not output == outputs[index]:
						error = outputs[index] - output
						self.update_weigths(perceptron, i,learning_ratio, error)
						#errors[index] = False
						errors = errors and False
					else:
						errors = errors and True
			epoca += 1
			print("\n\nErrors: ")
			print(errors)
			print("\n\n")

			if errors:
				print("Ending training - 100 %")
				input()
				break
			elif epoca == 1000:
				print("Atingiu o limite de épocas!")
				input()
				break

	def update_weigths(self, perceptron, inputs, learning_ratio, error):
		print("\n------------------------------------")
		print("UPDATING WEIGHTS")
		print("Error: ", error)
		print("Before weights: ", perceptron.weights)
		perceptron.weights = perceptron.weights + (learning_ratio * error * inputs)
		print("After weigths: ", perceptron.weights)
		print("---------------------------------------\n")

	def classify(self, input_t, threshold):
		print("\nClassifying input\n")
		perceptron = self.perceptrons[0]
		if perceptron.active_function(input_t, threshold) == 0:
			print(input_t , " -> Futebol")
		else:
			print(input_t , " -> Tenis")


if __name__ == "__main__":
	neural = NeuralNetwork(1)
	inputs = [[0,0], [0, 1], [1,0],[1,1]]
	outputs =[0, 0, 1, 1]
	learning_ratio = 0.6
	threshold = 0.4
	neural.training(inputs, outputs, learning_ratio, threshold)
	test  = [[1,1], [0,1], [1, 0], [0,0], [1,1]]
	for i in test:
		neural.classify(i,threshold)















