import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
import argparse
from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for evaluation
    parser.add_argument('--exp_name', type=str, help='model name')
    parser.add_argument('--dataset_name', type=str, help='model name', choices=['lasot', 'lasotext', 'tnl2k'])  

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    trackers = []
    dataset_name = 'LaSOT'

    args = parse_args()
    trackers.extend(trackerlist(name='UNINEXT', parameter_name=args.exp_name, dataset_name=dataset_name,
                                run_ids=None, display_name='UNINEXT'))

    # dataset = get_dataset('lasot') # "lasotext", "tnl2k" 
    dataset = get_dataset(args.dataset_name)
    # plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
    #              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
    print("Evaluating...")
    print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
    # print_per_sequence_results(trackers, dataset, "UNINEXT")
