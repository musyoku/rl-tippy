import json, os

def save_config(filename, config):
	config_dict = {}
	for key in dir(config):
		if key.startswith("_"):
			continue
		config_dict[key] = getattr(config, key)
	with open(filename, "w") as f:
		json.dump(config_dict, f, indent=4, sort_keys=True, separators=(',', ': '))

def load_config(filename, default, save_if_not_exist=True):
	config = default
	if filename is not None and os.path.isfile(filename):
		with open(filename, "r") as f:
			config_dict = json.load(f)
			for key, value in config_dict.items():
				setattr(config, key, value)
	elif save_if_not_exist:
		save_config(filename, config)
	return config