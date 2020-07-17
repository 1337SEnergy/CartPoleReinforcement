import json, time;
import numpy as np;
import keras, gym, agent;

def sign(value):
	return round(value/abs(value)) if value != 0 else 0;

def mean(values):
	return round(sum(values) / len(values), 2) if type(values) == list and len(values) > 0 else 0.0;

if __name__ == "__main__":
	env = gym.make("CartPole-v1");
	state_size = env.observation_space.shape[0];
	
	model_name = input("Model name -> ");
	load_trained = input("Load trained (y/n)? ");
	load_trained = load_trained.lower() == "y";
	
	my_model_location = "models/" + model_name + "/";
	my_model = my_model_location + ("model_trained.h5" if load_trained else "model.h5");

	epsilon = input("Epsilon -> ");
	
	print("Loading", my_model, "with epsilon", epsilon);
	agent = agent.DQNAgent(my_model);
	
	try: agent.memory = json.load(my_model_trained.replace(".h5", ".json"));
	except: agent.memory = [];

	episode_count = int(input("Episode count -> "));
	batch_size = 16;
	
	max_score = None;
	highest_score = 0;
	scores = [];
	rewards = [];
	
	start = time.time();
	first_start = start;
	
	for e in range(episode_count):		
		# at each episode, reset environment to starting position
		state = env.reset();
		state = np.reshape(state, [1, state_size]);
		score = 0;
		done = False;
		rewards.append(0.0);
		
		while not done and (score < max_score if max_score else True):
			# show game graphics
			# env.render();
			
			# select action, observe environment, calculate reward
			action = agent.act(state);
			next_state, reward, done, _ = env.step(action);
			next_state = np.reshape(next_state, [1, state_size]);
			score += 1;
			
			cart_distance = state[0][0];
			pole_angle = state[0][2];
			
			# try to keep the cart in the middle
			# decrease reward with distance from mid
			# decrease reward with bigger angle
			reward -= abs(cart_distance/0.6); # cart position, varies from -2.4 to 2.4
			reward -= abs(pole_angle/2); # pole angle in radians, varies from -0.21 to 0.21
			
			# save experience and update current state
			agent.remember(state, action, reward, next_state, done);
			state = next_state;
			rewards[-1] += reward;
			
			# dynamic batch_size and max_memory
			# batch_size = round((highest_score/500) * 80) + 48;
			# max_memory = round(highest_score*20 + 250) if highest_score != 500 else 9500;
			
			if len(agent.memory) > batch_size:
				agent.replay(batch_size);
		
		scores.append(score);
		if len(scores) > 100: scores = scores[-100:];
		if len(rewards) > 100: rewards = rewards[-100:];
		
		print("episode: {}/{}, score: {}, e: {:.2}, in memory: {}, last 100 average: {}"
				.format(e+1, episode_count, score, agent.epsilon, len(agent.memory), mean(scores)));
		
		if len(scores) >= 5 and score == 500 and sum(scores[-5:]) >= 1500:
			agent.save();
		
		if len(scores) >= 15 and sum(scores[-15:]) == 7500:
			print("training successfull!");
			agent.save("final");
			break;
			
		if score > highest_score: highest_score = score;
		
		if (e+1) % 5 == 0:
			print("Took", round((time.time()-start)/60, 2), "minutes\n");
			start = time.time();
			agent.merge_models();

	agent.save();
	print("Total training time:", round((time.time()-first_start)/60, 2), "minutes");