"""
Rede Neural que seja capaz de classificar se determnada
pessoa joga futebol ou tênis. As entradas devem ocorrer na 
sequência: Nadal(11), Gabriel(01), Federer(10), Nadal(00) 
e novamente Nadal(11) para verificar se a rede aprendeu.
"""
import numpy as np
from random import uniform

class Perceptron:
	def __init__(self, threshold):
		self.input = None
		self.output = None
		self.weights = None
		self.threshold = threshold

	def activation_function(self):
		result = (self.input * self.weights).sum()
		print(self.input )


		if result >= self.threshold:
			self.output = -1
		else: 
			self.output =  1
		print("\n------------------------------------")
		print("Activation Function(input * weights) = ",self.output)
		print("input: ", self.input)
		print("weights: ", self.weights)
		print("threshold: ", self.threshold)
		print("input * weights: ", self.input * self.weights)
		print("result: ", result)
		print("---------------------------------------\n")

		return self.output

	def set_input(self, input):
		self.input = np.array(input)

	def set_weights(self, weights):
		self.weights = np.array(weights)

	def set_threshold(self, threshold):
		self.threshold = threshold

	def update_weights(self, learning_ratio, error):
		print("\n------------------------------------")
		print("UPDATING WEIGHTS")
		print("Error: ", error)
		print("Before weights: ", self.weights)
		self.weights = self.weights + (learning_ratio * error * self.input)
		print("After weigths: ", self.weights)
		print("---------------------------------------\n")


class NeuralNetwork:
	def __init__(self, size, threshold):
		self.inputs = None
		self.outputs = None
		self.weights = None#[uniform(-1,1) for i in range(0, len(self.inputs[0])]
		self.learning_ratio = None
		self.perceptron = Perceptron(threshold)

	def training(self, inputs,  outputs, learning_ratio, weights=None):
		self.inputs = np.array(inputs)
		self.outputs = outputs
		self.learning_ratio = learning_ratio
		self.weights = [uniform(-0.5,0.5) for i in range(0, len(inputs[0]))]
		self.perceptron.set_weights(self.weights)
		self.epocas = 0

		print("===============TRAINING==================")
		print("Input: ", self.inputs)
		print("Desire Output: ", self.outputs)
		print("Learning Ratio: ", self.learning_ratio)
		print("Weights: ", self.weights)
		print("=========================================")

		while True:
			self.errors = np.array([False] * self.inputs.shape[0])
			for index, i in enumerate(self.inputs):
				self.perceptron.set_input(i)
				if not self.perceptron.activation_function() == self.outputs[index]:
					error = self.outputs[index] - self.perceptron.output
					print("Ai")
					print(error)
					self.perceptron.update_weights(learning_ratio, error)
					self.errors[index] = False
				else:
					self.errors[index] = True
			self.epocas += 1
			print("\n\nErrors: ")
			print(self.errors)
			print("\n\n")
			input()
			if np.all(self.errors):
				break

	def classify(self, input_t):
		self.perceptron.set_input(input_t)
		if self.perceptron.activation_function() == 1:
			print(input_t , " -> Futebol")
		else:
			print(input_t , " -> Tenis")


if __name__ == "__main__":
	neural = NeuralNetwork(1, 0.5)
	inputs = [[0,0], [0, 1], [1,0],[1,1]]
	outputs =[1, 1, -1, -1]
	learning_ratio = 0.1
	neural.training(inputs, outputs, learning_ratio)
	test  = [[1,1], [0,1], [1, 0], [0,0], [1,1]]
	for i in test:
		neural.classify(i)













