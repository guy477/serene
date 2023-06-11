
from poker.poker_game import PokerGame
from poker.cfr import CFRTrainer
import poker.ccluster as ccluster

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.special import comb
from sklearn.cluster import MiniBatchKMeans

def main():
    num_players = 2
    num_ai_players = 2

    # pot relative bet-sizings for preflop, flop, turn, and river
    bet_sizing = [(2.5, 8), (.33, .70), (.40, .82, 1.2), (.75, 1.2, 2)]

    initial_chips = 1000
    small_blind = 5
    big_blind = 10

    num_hands = 5

    num_simulations = 250

    num_iterations = 25
    realtime_iterations = 50
    cfr_depth = 1000
    cfr_realtime_depth = 1000
    

    # Train the AI player using the CFR algorithm
    cfr_trainer = CFRTrainer(num_iterations, realtime_iterations, num_simulations, cfr_depth, cfr_realtime_depth, num_players, initial_chips, small_blind, big_blind, bet_sizing)
    strategy_list = cfr_trainer.train()
    plot_hands(strategy_list)

    game = PokerGame(num_players, initial_chips, num_ai_players, small_blind, big_blind, bet_sizing, cfr_trainer)

    # Play the game
    game.play_game(num_hands)
    print('\n\n')

def cluster():
    # This is only accurate for k = 5; n = 52. All other combinations should be used solely for testing
    # unless you know what you're doing.
    k = 5
    n = 24
    rvr_clstrs = 40
    trn_clstrs = 80
    flp_clstrs = 90

    precompute = True


    # Enter the number of threads for you system
    threads = 8


    # If this is your first time running, make sure new_file is set to true.
    # This has only been tested on linux systems.
    # Be sure your 'results/' directory has > 70G
    # and that you have either 128G ram or 100G+ swap.


    # Unless you're running this program on an excess of 32 threads, using swap
    # memory off an SSD should not be a bottleneck for this part of the program.
    # OS LIMITATIONS MIGHT CAUSE BOTTLENECKS WHEN DUMPING TO AN EXTERNAL SDD/HDD
    # BE SURE TO RUN THE PROGRAM FROM THE SAME DRIVE THAT CONTAINS YOUR EXTENDED
    # SWAP MEMORY. IF YOU HAVE ENOUGH RAM, DONT WORRY BOUT IT.


    # #########################################################################################################
    # #########################################################################################################



    t = time.time()
    print('computing river ehs')
    dupes = ccluster.river_ehs(n, k, threads, new_file = True)
    print('Time spent: ' + str((time.time() - t)/60) + 'm')

    #                      Load a memory mapping of the river scores.

    z = np.memmap('results/river_f.npy', mode = 'c', dtype = np.float32, shape = (int(comb(n, 2)) * (int(comb(n, k)) - dupes), 1))

    # #########################################################################################################
    # #########################################################################################################

    # we will also need to modify a kmeans clustering algorithm to use memmapped objects to avoid memory prob


    # highly recommended.
    precompute = True

    if precompute:

        t = time.time()

        print('precomputing river centers')
        centers = ccluster.kmc2(z, rvr_clstrs, chain_length=50, afkmc2=True)
        np.save('results/cntrs_RIVER', centers)
        print('Time spent' + str((time.time() - t)/60) + 'm')
    else:
        centers = None


    # #########################################################################################################
    # #########################################################################################################


    t = time.time()
    print('computing river clusters')
    miniK = MiniBatchKMeans(n_clusters = rvr_clstrs, batch_size=(rvr_clstrs//2)*(threads * 265), tol = 10e-8, max_no_improvement = None, init = centers, verbose=False, n_init=1).fit(z)
    print('Time spent: ' + str((time.time() - t)/60) + 'm')

    # #########################################################################################################
    # #########################################################################################################

    np.save('results/adjcntrs_RIVER', miniK.cluster_centers_)
    np.save('results/lbls_RIVER', miniK.labels_)


    adjcntrs = np.load('results/adjcntrs.npy', mmap_mode = 'r')
    lbls = np.load('results/lbls.npy', mmap_mode = 'r')

    # #########################################################################################################
    # #########################################################################################################

    #print(adjcntrs)

    #########################################################################################################
    # print('Saving River')
    # pd.DataFrame(z[:int(comb(n, k)) - dupes]).to_csv('river_ehs_23s(24cards).csv')
    # df = pd.DataFrame(list(map(lambda x: adjcntrs[x], lbls[:int(comb(n, k)) - dupes])))
    # df.to_csv('river_ehs_clst_23s.csv')
    # print("saved")
    #########################################################################################################


    # #########################################################################################################
    # TURN #########################################################################################################
    # #########################################################################################################
    print('\n\n')

    t = time.time()
    print('computing turn ehs')
    dupes = ccluster.turn_ehs(n, k, 16, True)
    print('Time spent: ' + str((time.time() - t)/60) + 'm')



    turn_prob_dist = np.memmap('results/prob_dist_TURN.npy', mode = 'c', dtype = np.float32, shape = (int(comb(n, 2)) * ((int(comb(n, 4)) - dupes)), n - k - 2))
    print(turn_prob_dist)


    # #########################################################################################################
    # #########################################################################################################


    if precompute:
        t = time.time()
        
        print('precomputing turn centers')
        centers_TURN = ccluster.kmc2(turn_prob_dist, trn_clstrs, chain_length=50, afkmc2=True)
        np.save('results/cntrs_TURN', centers_TURN)

        print('Time spent precalculating centers: ' + str((time.time() - t)/60) + 'm')
    else:
        centers_TURN = None


    # #########################################################################################################
    # #########################################################################################################

    t = time.time()
    print('computing turn clusters')
    miniK = MiniBatchKMeans(n_clusters = trn_clstrs, batch_size=(trn_clstrs//2)*(threads * 265), tol = 10e-8, max_no_improvement = None, init = centers_TURN, verbose=False, n_init=1).fit(turn_prob_dist)
    print('Time spent: ' + str((time.time() - t)/60) + 'm')


    np.save('results/adjcntrs_TURN', miniK.cluster_centers_)
    np.save('results/lbls_TURN', miniK.labels_)

    adjcntrs = np.load('results/adjcntrs_TURN.npy', mmap_mode = 'r')
    lbls = np.load('results/lbls_TURN.npy', mmap_mode = 'r')

    #print(adjcntrs)


    #########################################################################################################
    # print('Saving Turn')
    # pd.DataFrame(turn_prob_dist[:int(comb(n, k-1)) - dupes]).to_csv('turn_prob_dist_23s(24cards).csv')
    # pd.DataFrame(list(map(lambda x: adjcntrs[x], lbls[:int(comb(n, k-1)) - dupes]))).to_csv('turn_prob_dist_clst_23s.csv')
    # print("saved")
    #########################################################################################################


    # #########################################################################################################
    # FLOP #########################################################################################################
    # #########################################################################################################
    print('\n\n')


    t = time.time()
    print('computing flop ehs')
    dupes = ccluster.flop_ehs(n, k, 16, True)
    print('Time spent: ' + str((time.time() - t)/60) + 'm')


    flop_prob_dist = np.memmap('results/prob_dist_FLOP.npy', mode = 'c', dtype = np.float32, shape = (int(comb(n, 2)) * ((int(comb(n, 3)) - dupes)), n - k - 1))

    print(flop_prob_dist)
    #########################################################################################################
    #########################################################################################################


    #########################################################################################################


    if precompute:
        t = time.time()
        
        print('precomputing centers ---- FLOP')
        centers_FLOP = ccluster.kmc2(flop_prob_dist, flp_clstrs, chain_length=50, afkmc2=True)
        np.save('results/cntrs_FLOP', centers_FLOP)

        print('Time spent precalculating centers: ' + str((time.time() - t)/60) + 'm')
    else:
        centers_FLOP = None

    t = time.time()
    print('computing flop clusters')
    miniK = MiniBatchKMeans(n_clusters = flp_clstrs, batch_size=(flp_clstrs//2)*(threads * 265), tol = 10e-8, max_no_improvement = None, init = centers_FLOP, verbose=False, n_init=1).fit(flop_prob_dist)
    print('Time spent: ' + str((time.time() - t)/60) + 'm')


    np.save('results/adjcntrs_FLOP', miniK.cluster_centers_)
    np.save('results/lbls_FLOP', miniK.labels_)


    adjcntrs = np.load('results/adjcntrs_FLOP.npy', mmap_mode = 'r')
    lbls = np.load('results/lbls_FLOP.npy', mmap_mode = 'r')

    #print(adjcntrs)

    #########################################################################################################
    # print('Saving Flop')
    # pd.DataFrame(flop_prob_dist[:int(comb(n, k-2)) - dupes]).to_csv('flop_prob_dist_23s(24cards).csv')
    # pd.DataFrame(list(map(lambda x: adjcntrs[x], lbls[:int(comb(n, k-2)) - dupes]))).to_csv('flop_prob_dist_clst_23s.csv')
    # print("saved")
    #########################################################################################################


def plot_hands(strategy_list):
    strategy_df = pd.DataFrame(strategy_list)

    # Rename the columns for clarity
    strategy_df.columns = ['Position', 'Hand', 'Strategy']
    strategy_df.sort_values(by = 'Hand')
    strategy_df = strategy_df[(strategy_df['Position'] == 'SB') & (strategy_df['Strategy']!={})]
    strategy_df.reset_index(inplace = True)
    # Create a new DataFrame to store the individual strategy proportions
    strategy_proportions = pd.DataFrame()

    # Split the Strategy dictionary into separate columns in strategy_proportions
    print(strategy_df)
    
    for index, row in strategy_df.iterrows():
        for action, value in row.Strategy.items():
            strategy_proportions.at[index, str(action)] = value
    
    # Fill any missing values with 0
    strategy_proportions.fillna(0, inplace=True)
    
    # Reset the index to make it easier to plot
    strategy_proportions.reset_index(drop=True, inplace=True)
    
    
    subplot_size = np.array([.5, .5])
    # Number of columns for subplot grid
    ncols = 13
    # Number of rows for subplot grid
    nrows = 13

    figsize = subplot_size * [ncols, nrows]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    # Set up colors for the actions. You can adjust this to your liking.
    colors = plt.cm.get_cmap('tab10', len(strategy_df['Strategy'].iloc[0]))

    for i, (idx, row) in enumerate(strategy_proportions.iterrows()):
        ax = axes[i]
        ax.set_title(strategy_df.loc[idx, 'Hand'], size=5, pad=-500)
        ax.set_ylim([0, 1]) # since they are proportions
        ax.get_xaxis().set_visible(False)  # remove the x-axis
        ax.get_yaxis().set_visible(False)  # remove the y-axis

        bottom = 0 # Initialize the bottom of the bars to 0
        for j in range(len(row.index)):
            ax.bar(i, row.values[j], bottom=bottom, color=colors(j))
            bottom += row.values[j]  # Add the height of the bar to the bottom for the next bar


    # Remove the unused subplot, if any
    if len(strategy_proportions.index) % ncols != 0:
        for j in range(i+1, ncols * nrows):
            fig.delaxes(axes[j])

    # Create legend for the whole figure
    patches = [mpatches.Patch(color=colors(i), label=strategy_proportions.columns[i]) for i in range(len(strategy_proportions.columns))]
    plt.legend(handles=patches, loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.4)  # adjust the space between plots
    plt.show()


if __name__ == "__main__":
    # cluster()
    main()