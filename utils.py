import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

        self.states_size = None  # np.zeros(self.batchSize,  dtype='int32')
        self.states_batch = (
            None  # np.zeros((self.batchSize, self.maxServiceLength),  dtype='int32')
        )
        self.len_mask = None

    def generate_states_batch(self, batch_size=None):
        """Generate new batch of initial states"""
        if batch_size is None:
            batch_size = self.batch_size
        items_seqs_batch = np.random.randint(
            low=self.min_item_size,
            high=self.max_item_size + 1,
            size=(batch_size, self.max_num_items),
        )
        items_len_mask = np.ones_like(items_seqs_batch, dtype="int32")
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


def plot_training_history(
    config,
    data: list[list[float]],
    labels: list[str],
    moving_avg_window: int = 0,
    outfilepath: str = "./train_hist.png",
):
    plt.style.use("./tseries.mplstyle")
    fig, line_ax = plt.subplots(1, 1)

    # Set titles and subtitles
    line_ax.set_title(f"Average Occupancy Ratio (%) (moving average n={moving_avg_window})")
    subtitle = (
        f"Problem conditions: {config.min_num_items} <= # of items "
        f"<= {config.max_num_items} | {config.min_item_size} <= item size <= {config.max_item_size} | "
        f"bin size = {config.bin_size}"
    )

    line_ax.text(x=0, y=1.03, s=subtitle, transform=line_ax.transAxes)

    # Add line data and format it
    for line, label in zip(data, labels):
        if moving_avg_window > 0:
            line = np.convolve(line, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
            line = np.insert(line, 0, [None]*(moving_avg_window-2))
        
        if 'DRL' in label:
            line_ax.plot(range(len(line)), line, label=label)
        else:
            line_ax.plot(range(len(line)), line, label=label, linestyle='dashed')


    # Dynamically set Y axis floating points
    line_ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(f"{{x:.0%}}"))
    if len(data) > 1:
        line_ax.legend(bbox_to_anchor=(0.8, 0.5))
    max_occ = max([np.max(line) for line in data])
    min_occ = min([np.min(line) for line in data])
    line_ax.set_ylim(min_occ - 0.03, max_occ + 0.01)

    # Set X labels
    line_ax.set_xlabel("Training steps")
    x_ticks = np.arange(0, config.n_episodes + 1, config.n_episodes // 4)
    line_ax.set_xticks(x_ticks)
    line_ax.set_xticklabels(x_ticks)


    # Save figure
    fig.savefig(outfilepath)
