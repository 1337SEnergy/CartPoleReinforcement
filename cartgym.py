import json, time;
import numpy as np;
import keras, gym, agent;

def sign(val):
	return round(val/abs(val)) if val != 0 else 0;

def mean(vals):
	return round(sum(vals) / len(vals), 2) if type(vals) == list else 0.0;

if __name__ == "__main__":
	env = gym.make("CartPole-v1");
	state_size = env.observation_space.shape[0];
	agent = agent.CartPoleAgent(env.action_space.n);
	
	modelName = input("Model name -> ");
	loadTrained = input("Load trained (y/n)? ");
	loadTrained = True if loadTrained.lower() == "y" else False;
	
	myModelTrained = "models/" + modelName + "/model_trained.h5";
	myModel = myModelTrained if loadTrained else "models/" + modelName + "/model.h5";

	epsilon = input("Epsilon -> ") if loadTrained else 1.0;
	
	print("Loading", myModel, "with epsilon", epsilon);
	agent.load(myModel, float(epsilon));

	episodeCount = int(input("Episode count -> "));
	batch_size = 64;
	max_memory = 15000;
	
	maxScore = None;
	highestScore = 0;
	scores = [];
	
	start = time.time();
	firstStart = start;
	
	for e in range(episodeCount):		
		# at each episode, reset environment to starting position
		state = env.reset();
		state = np.reshape(state, [1, state_size]);
		score = 0;
		done = False;
		
		while not done and (score < maxScore if maxScore else True):
			# show game graphics
			env.render();

			# select action, observe environment, calculate reward
			action = agent.act(state);
			next_state, reward, done, _ = env.step(action);
			next_state = np.reshape(next_state, [1, state_size]);
			score += 1;
			
			# try to keep the cart in the middle
			# decrease reward with distance from mid
			# decrease reward with bigger angle
			# reward -= abs(state[0][0]/9.6); # cart position, varies from -2.4 to 2.4
			# reward -= abs(state[0][2]); # pole angle in radians, varies from -0.21 to 0.21
			
			cartDistance = state[0][0];
			cartSpeed = state[0][1];
			poleAngle = state[0][2];
			
			# cart is not in the middle
			if abs(cartDistance) > 0.3:
				reward -= abs(cartDistance)/2.4;
				# yet still going further from the middle
				# if sign(cartDistance) == sign(cartSpeed):
					# reward -= min((abs(cartDistance)/2.4) * max(abs(cartSpeed), 0.1), 1.0);
			
			# cart is in the middle
			# decrease reward based on the pole angle
			else:
				reward -= abs(poleAngle/2);
			
			# save experience and update current state
			agent.remember(state, action, reward, next_state, done);
			state = next_state;
			
			# dynamic batch_size and max_memory
			# batch_size = round((highestScore/500) * 80) + 48;
			# max_memory = round(highestScore*20 + 250) if highestScore != 500 else 9500;
			
			if len(agent.memory) > batch_size:
				agent.replay(batch_size, max_memory);
		
		scores.append(score);
		if len(scores) > 100: scores = scores[-100:];
		
		print("episode: {}/{}, score: {}, e: {:.2}, in memory: {}, highest score: {}, batch size: {}, last 100 average: {}"
				.format(e+1, episodeCount, score, agent.epsilon, len(agent.memory), highestScore, batch_size, mean(scores)));
		
		if score >= highestScore:
			agent.save(myModelTrained);
			highestScore = score;
		
		if (e+1) % 5 == 0:
			print("Took", round((time.time()-start)/60, 2), "minutes\n");
			start = time.time();

	agent.save(myModelTrained);
	print("Total training time:", round((time.time()-firstStart)/60, 2), "minutes");