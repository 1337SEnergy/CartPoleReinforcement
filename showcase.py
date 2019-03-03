import json, time;
import numpy as np;
import keras, gym, agent;

def mean(lst):
	return round(sum(lst) / len(lst), 2);

if __name__ == "__main__":
	env = gym.make("CartPole-v1");
	state_size = env.observation_space.shape[0];
	action_size = env.action_space.n;
	agent = agent.CartPoleAgent(action_size);
	
	modelName = input("Model name -> ");
	myModel = "models/" + modelName + "/model_trained.h5";
	epsilon = input("Epsilon -> ");
	
	print("Loading", myModel, "with epsilon", epsilon);
	agent.load(myModel, float(epsilon));

	episodeCount = int(input("Episode count -> "));
	done = False;
	
	highestScore = 0;
	scores = [];
	
	start = time.time();
	firstStart = start;
	
	for e in range(episodeCount):		
		# at each episode, reset environment to starting position
		state = env.reset();
		state = np.reshape(state, [1, state_size]);
		score = 0;
		
		while not done:
			# show game graphics
			env.render();

			# select action, observe environment, calculate reward
			action = agent.act(state);
			state, reward, done, _ = env.step(action);
			state = np.reshape(state, [1, state_size]);
			
			score += 1;
		
		done = False;
		scores.append(score);
		if len(scores) > 100: scores = scores[-100:];
		
		print("episode: {}/{}, score: {}, e: {:.2}, highest score: {}, last 100 average: {}"
				.format(e+1, episodeCount, score, agent.epsilon, highestScore, mean(scores)));
		
		if score >= highestScore:
			highestScore = score;
		
		if (e+1) % 5 == 0:
			print("Took", round((time.time()-start)/60, 2), "minutes\n");
			start = time.time();

	print("Showcase time:", round((time.time()-firstStart)/60, 2), "minutes");