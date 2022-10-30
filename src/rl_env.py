"""
Logic to generate new states and compute the reward for each state-action pair.
"""

import numpy as np
from torch import dtype

class StatesGenerator(object):
    """
    Helper class used to randomly generate batches of states given a set
    of problem conditions, which are provided via the `config` object.
    """

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.min_num_items = config.min_num_items
        self.max_num_items = config.max_num_items
        self.min_item_size = config.min_item_size
        self.max_item_size = config.max_item_size

    def generate_states_batch(self, batch_size=None):
        """Generate new batch of initial states"""
        if batch_size is None:
            batch_size = self.batch_size
        items_seqs_batch = np.random.randint(
            low=self.min_item_size,
            high=self.max_item_size + 1,
            size=(batch_size, self.max_num_items),
        )
        items_len_mask = np.ones_like(items_seqs_batch, dtype="float32")
        items_seq_lens = np.random.randint(
            low=self.min_num_items, high=self.max_num_items + 1, size=batch_size
        )
        for items_seq, len_mask, seq_len in zip(
            items_seqs_batch, items_len_mask, items_seq_lens
        ):
            items_seq[seq_len:] = 0
            len_mask[seq_len:] = 0

        return (
            items_seqs_batch,
            items_seq_lens,
            items_len_mask,
        )

def avg_occupancy(
    bin_size: int, items_size: tuple[float], items_order: tuple[int], heuristic: str
) -> float:
    """
    Calculates the average occupancy of the used bins given:
    - `bin_size`: the bin size (assumed constantfor all bins)
    - `items_size`: the size of the items to be allocated
    - `items_order`: the order in which the items are allocated (this is the output of the
    pointer network)
    - `heuristic`: the heuristic used to allocate the items in the bins given the order.
      It can be either "NF" (next fit) or "FF" (first fit)
    
    """
    if heuristic not in ("NF", "FF"):
        raise ValueError(f"Unknown heuristic: {heuristic}")
    bins = [0]
    for item_idx in items_order:
        if item_idx == -1: # sequence is shorter than max_num_items 
            continue
        item_size = items_size[item_idx]
        if heuristic == "NF":
            if bins[-1] + item_size <= bin_size:
                bins[-1] += item_size
            else:
                bins.append(item_size)
        elif heuristic == "FF":
            for bin_idx, bin_occupancy in enumerate(bins):
                if bin_occupancy + item_size <= bin_size:
                    bins[bin_idx] += item_size
                    break
            else:
                bins.append(item_size)

    return np.mean(np.array(bins) / bin_size)

def compute_reward(config, states_batch, len_mask, actions_batch):
    """
    Compute the average occupancy ratio for each state-action pair in the batch.
    """
    bin_size = config.bin_size
    # states_batch = states_batch.squeeze(-1).numpy()
    actions_batch = actions_batch.numpy().astype(int)
    avg_occupancy_ratios = []
    for states, actions in zip(states_batch, actions_batch):
        avg_occupancy_ratios.append(avg_occupancy(bin_size, states, actions, heuristic=config.agent_heuristic))

    return np.array(avg_occupancy_ratios)

def get_benchmark_rewards(config, states_generator: StatesGenerator=None, states_batch=None):
    """
    Compute the average occupancy ratio following the NF, FF and FFD heuristics. 
    
    If the arg `states_generator` is provided, the states (i.e. sequences of items to allocate) are randomly
    generated during 1,000 loops and the retruend values are the average.
    If no `states_generator`arg is provided and a `states_batch` arg is provided, then the average 
    occupancy ratio is computed for the provided batch.

    Returns a tuple with 3 values corresponding to the average occupancy ratio obtained following
    a NF, FF and FFD heuristic respectively.
    """
    nf_reward, ff_reward, ffd_reward = [], [], []
    if states_generator is not None:
        states, states_lens, len_mask = states_generator.generate_states_batch(
            batch_size=10000
        )
    else:
        states = states_batch

    items_order_default = np.arange(config.max_num_items)
    for state in states:

        nf_reward.append(
            avg_occupancy(config.bin_size, state, items_order_default, heuristic="NF")
        )
        ff_reward.append(
            avg_occupancy(config.bin_size, state, items_order_default, heuristic="FF")
        )
        items_order_decreasing = np.flip(np.argsort(state))
        ffd_reward.append(
            avg_occupancy(config.bin_size, state, items_order_decreasing, heuristic="FF")
        )

    return np.mean(nf_reward), np.mean(ff_reward), np.mean(ffd_reward)