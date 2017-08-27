import json, os

class Config(object):
	def __init__(self):
		self.agent_history_length = 	4
		self.agent_action_frequency = 	4
		self.replay_memory_size = 		10 ** 6
		self.replay_start_size = 		5 * 10 ** 4
		self.target_update_frequency =	10 ** 4
		self.discount_factor = 			0.99
		self.initial_exploration_rate =	1.0
		self.final_exploration_rate =	0.1
		self.final_exploration_frame = 	10 ** 6
		self.no_op_max = 				30
		self.grad_clip =				1	
		self.weight_decay =				1e-6	
		self.initial_learning_rate =	0.001	
		self.lr_decay =					1	
		self.momentum =					0.9	
		self.optimizer =				"adam"	
		self.batchsize =	 			32

def save_config(filename, config):
	config_dict = {}
	for key in dir(config):
		if key.startswith("_"):
			continue
		config_dict[key] = getattr(config, key)
	with open(filename, "w") as f:
		json.dump(config_dict, f, indent=4, sort_keys=True, separators=(',', ': '))

def load_config(filename, save_if_not_exist=True):
	config = Config()
	if filename is not None and os.path.isfile(filename):
		with open(filename, "r") as f:
			config_dict = json.load(f)
			for key, value in config_dict.items():
				setattr(config, key, value)
	elif save_if_not_exist:
		save_config(filename, config)
	return config