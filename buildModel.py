import keras, json;
import numpy as np;

# NN
def CreateModel(layers, learning_rate):
	if type(layers) != list: return None;
	
	model = keras.models.Sequential();
	
	if len(layers) == 0:
		model.add(keras.layers.Dense(2, input_dim=4, activation="linear", name=("lr_{}".format(learning_rate))));
	else:
		model.add(keras.layers.Dense(layers[0], input_dim=4, name=("lr_{}".format(learning_rate))));
		model.add(keras.layers.Activation("tanh"));
		# model.add(keras.layers.LeakyReLU(alpha=0.3));
	
		for neuronAmount in layers[1:]:
			model.add(keras.layers.Dense(neuronAmount));
			model.add(keras.layers.Activation("tanh"));
			# model.add(keras.layers.LeakyReLU(alpha=0.3));
		
		model.add(keras.layers.Dense(2, activation="linear"));
		
	model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=["accuracy"]);
	return model;

modelName = input("Model name -> ");
layers = json.loads(input("Neurons in hidden layers as list (e.g. [6, 3, 3]) -> "));
lr = float(input("Learning rate -> "));

model = CreateModel(layers, lr);
if not model: raise Exception("Invalid model, did you use correct param format?");

dirName = "models/" + modelName + "/";
import os;
if not os.path.exists(dirName):
	os.makedirs(dirName);

model.summary();
keras.utils.plot_model(model, show_shapes=True, to_file=dirName + "model.png");
model.save(dirName + "model.h5");