import numpy as np


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
    states_batch = states_batch.squeeze(-1).numpy()
    actions_batch = actions_batch.numpy()
    avg_occupancy_ratios = []
    for states, actions in zip(states_batch, actions_batch):
        avg_occupancy_ratios.append(avg_occupancy(bin_size, states, actions, heuristic='NF'))

    return np.array(avg_occupancy_ratios)


def get_benchmark_rewards(config, states_generator):
    nf_reward, ff_reward, ffd_reward = [], [], []
    items_order_default = np.arange(config.max_num_items)
    for _ in range(10000):
        states, states_lens, len_mask = states_generator.generate_states_batch(
            batch_size=1
        )

        nf_reward.append(
            avg_occupancy(config.bin_size, states[0], items_order_default, heuristic="NF")
        )
        ff_reward.append(
            avg_occupancy(config.bin_size, states[0], items_order_default, heuristic="FF")
        )
        items_order_decreasing = np.flip(np.argsort(states[0]))
        ffd_reward.append(
            avg_occupancy(config.bin_size, states[0], items_order_decreasing, heuristic="FF")
        )

    return np.mean(nf_reward), np.mean(ff_reward), np.mean(ffd_reward)