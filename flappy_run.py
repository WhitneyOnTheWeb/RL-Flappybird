import sys
sys.path.append('/game')
sys.path.append('/learner')
import learner.agent as RL

agent = RL.Agent(state_size           = 16,       # number of frames to stack
                 frames_per_action    = 2,                 
                 max_games            = 5000,    # number times to play game
                 fps                  = 30,      # frames per second
                 max_game_minutes     = 10,                 
                 game_score_target    = 10,      # goal
                 keep_gif_for_score   = 3,       # game score bar to keep GIF
                 initial_epsilon      = 0.1,     # start random action probability
                 terminal_epsilon     = 0.0001,  # end random action probability
                 observation_steps    = 5000,    # steps pre training
                 exploration_steps    = 500000,  # epsilon decay duration
                 save_every_n_steps   = 10000,   # save model every n steps
                 training             = False,   # observe then train if False
                 learn_rate           = 0.001,   # learning rate
                 states_in_memory     = 1000000, # replay memory size
                 training_sample_size = 32,      # experience sample size
)

agent.play()