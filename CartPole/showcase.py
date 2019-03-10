import json, time;
import numpy as np;
import keras, gym, agent;

def mean(values):
	return round(sum(values) / len(values), 2) if type(values) == list and len(values) > 0 else 0.0;

if __name__ == "__main__":
	env = gym.make("CartPole-v1");
	state_size = env.observation_space.shape[0];
	agent = agent.CartPoleAgent(env.action_space);
	
	model_name = input("Model name -> ");
	my_model = "models/" + model_name + "/model_trained.h5";
	epsilon = input("Epsilon -> ");
	
	print("Loading", my_model, "with epsilon", epsilon);
	agent.load(my_model, float(epsilon));

	episode_count = int(input("Episode count -> "));
	done = False;
	
	max_score = None;
	highest_score = 0;
	scores = [];
	
	start = time.time();
	first_start = start;
	
	for e in range(episode_count):		
		# at each episode, reset environment to starting position
		state = env.reset();
		state = np.reshape(state, [1, state_size]);
		score = 0;
		done = False;
		
		while not done and (score < max_score if max_score else True):
			# show game graphics
			env.render();

			# select action, observe environment, calculate reward
			action = agent.act(state);
			state, reward, done, _ = env.step(action);
			state = np.reshape(state, [1, state_size]);
			
			score += 1;
		
		scores.append(score);
		if len(scores) > 100: scores = scores[-100:];
		
		print("episode: {}/{}, score: {}, e: {:.2}, highest score: {}, last 100 average: {}"
				.format(e+1, episode_count, score, agent.epsilon, highest_score, mean(scores)));
		
		if score >= highest_score:
			highest_score = score;
		
		if (e+1) % 5 == 0:
			print("Took", round((time.time()-start)/60, 2), "minutes\n");
			start = time.time();

	print("Showcase time:", round((time.time()-first_start)/60, 2), "minutes");