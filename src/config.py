import argparse

parser = argparse.ArgumentParser()
arg_lists = []

def str2bool(v):
    return v.lower() in ("true", "1")

parameters_definition = {

    # PROBLEM CONDITIONS #
    "min_item_size": { "value": 3, "type": int, "desc": "Minimum item size"},
    "max_item_size": { "value": 8, "type": int, "desc": "Maximum item size"},
    "min_num_items": { "value": 40, "type": int, "desc": "Minimum number of items"},
    "max_num_items": { "value": 40, "type": int, "desc": "Maximum number of items"},
    "bin_size": { "value": 10, "type": int, "desc": "Bin size"},
    "agent_heuristic": {"value": "FF", "type": str, "desc": "Heursitic used by the agent to allocate the sequence output"},

    # TRAINING PARAMETERS #
    "seed": { "value": 3, "type": int, "desc": "Random seed"},
    "n_episodes": { "value": 5000, "type": int, "desc": "Number of episodes"},
    "batch_size": { "value": 128, "type": int, "desc": "Batch size"},
    "lr": { "value": 1.0e-3, "type": float, "desc": "Initial learning rate"},

    # NETWORK PARAMETERS #
    "hid_dim": { "value": 64, "type": int, "desc": "Hidden dimension"},

    # RUN OPTIONS #
    "device": { "value": "cpu", "type": str, "desc": "Device to use (if no GPU available, value should be 'cpu')"},
    "inference": {"value": False, "type": str2bool, "desc": "Do not train the model"},
    "model_path": {"value": None, "type": str, "desc": "Path to the model checkpoint to save if in training mode, or to load if in inference mode"},
}

def get_config():
    parser = argparse.ArgumentParser()
    for arg, arg_def in parameters_definition.items():
        parser.add_argument(f"--{arg}", type=arg_def["type"], default=arg_def["value"], help=arg_def["desc"])
    config, unparsed = parser.parse_known_args()
    return config, unparsed
