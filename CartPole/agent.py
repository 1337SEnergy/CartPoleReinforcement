import keras, random, json;
import numpy as np;

class CartPoleAgent(object):
	def __init__(self, action_space, model=None, epsilon=1.0):
		self.action_space = action_space;
		self.location = None;
		self.model = model;
		self.memory = [];
		
		self.gamma = 1.0; # long term value
		self.epsilon = epsilon; # how random are our choices?
		self.epsilon_min = 0.05;
		self.epsilon_decay = 0.997;

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done));
		
	def act(self, state):
		if random.random() <= self.epsilon:
			return self.action_space.sample();
	
		act_values = self.model.predict(state);
		return np.argmax(act_values[0]);

	def replay(self, batch_size, max_memory):
		max_memory = max(max_memory, 1);		
		if len(self.memory) > max_memory: self.memory = self.memory[-max_memory:];
		
		# select random samples from memory
		batch_size = min(batch_size, len(self.memory));
		minibatch = random.sample(self.memory, batch_size);
		
		xtrain = [];
		ytrain = [];
		for state, action, reward, next_state, done in minibatch:
			# compute new target value to be the output of the network
			target = reward if done else (reward + self.gamma *
						  np.amax(self.model.predict(next_state)[0]));
			
			# get current output value and update the value for the action
			target_f = self.model.predict(state)[0];
			target_f[action] = target;
			
			xtrain.append(state[0]);
			ytrain.append(target_f);
		
		if len(xtrain) > 0 and len(xtrain) == len(ytrain):
			xtrain = np.array(xtrain, dtype=np.float32);
			ytrain = np.array(ytrain, dtype=np.float32);
			self.model.fit(xtrain, ytrain, epochs=1, batch_size=batch_size, verbose=0);
		
		# update epsilon to minimize exploration on the long term
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay;
	
	def load(self, model_location, epsilon=1.0):
		self.model = keras.models.load_model(model_location);
		self.location = model_location;
		
		self.epsilon = max(epsilon, 0.0);
		self.epsilon = min(self.epsilon, 1.0);
	
	def save(self, model_location=None):
		try:
			location = self.location.replace(".h5", "_trained.h5") if not model_location else model_location + "model_trained.h5";
			
			self.model.save(location);
		
			# with open(location.replace(".h5", ".json"), "w+") as f:
				# json.dump([list(x) for x in self.memory], f);
			
		except Exception as e:
			print("Exception", str(e));