
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

def _6_max_opening():
    # Unopened ranges (Early to Late)
    UTG_OPEN = []
    MP_OPEN = [('fold', 0)]  # Assuming UTG has folded
    CO_OPEN = [('fold', 0), ('fold', 0)]  # Assuming UTG and MP have folded
    BTN_OPEN = [('fold', 0), ('fold', 0), ('fold', 0)]  # Assuming UTG, MP, and CO have folded
    SB_OPEN = [('fold', 0), ('fold', 0), ('fold', 0), ('fold', 0)]  # Assuming UTG, MP, CO, and BTN have folded

    # Defensive ranges (Early vs. Early to Late vs. Late)
    MP_DEF = [('raise', 1.5)]  # Assuming UTG has raised
    CO_UTG_DEF = [('raise', 1.5), ('fold', 0)]  # Assuming UTG has raised, MP has folded
    CO_MP_DEF = [('fold', 0), ('raise', 1.5)]  # Assuming UTG has folded, MP has raised
    BTN_UTG_DEF = [('raise', 1.5), ('fold', 0), ('fold', 0)]  # Assuming UTG has raised, MP and CO have folded
    BTN_MP_DEF = [('fold', 0), ('raise', 1.5), ('fold', 0)]  # Assuming UTG has folded, MP has raised, CO has folded
    BTN_CO_DEF = [('fold', 0), ('fold', 0), ('raise', 1.5)]  # Assuming UTG and MP have folded, CO has raised
    SB_UTG_DEF = [('raise', 1.5), ('fold', 0), ('fold', 0), ('fold', 0)]  # Assuming UTG has raised, MP, CO, and BTN have folded
    SB_MP_DEF = [('fold', 0), ('raise', 1.5), ('fold', 0), ('fold', 0)]  # Assuming UTG has folded, MP has raised, CO and BTN have folded
    SB_CO_DEF = [('fold', 0), ('fold', 0), ('raise', 1.5), ('fold', 0)]  # Assuming UTG and MP have folded, CO has raised, BTN has folded
    SB_BTN_DEF = [('fold', 0), ('fold', 0), ('fold', 0), ('raise', 1.5)]  # Assuming UTG, MP, and CO have folded, BTN has raised

    BB_DEF = [('raise', 1.5), ('fold', 0), ('fold', 0), ('fold', 0), ('fold', 0)]  # Assuming UTG has raised, MP, CO, BTN, and SB have folded
    BB_MP_DEF = [('fold', 0), ('raise', 1.5), ('fold', 0), ('fold', 0), ('fold', 0)]  # Assuming UTG has folded, MP has raised, CO, BTN, and SB have folded
    BB_CO_DEF = [('fold', 0), ('fold', 0), ('raise', 1.5), ('fold', 0), ('fold', 0)]  # Assuming UTG and MP have folded, CO has raised, BTN and SB have folded
    BB_BTN_DEF = [('fold', 0), ('fold', 0), ('fold', 0), ('raise', 1.5), ('fold', 0)]  # Assuming UTG, MP, and CO have folded, BTN has raised, SB has folded
    BB_SB_DEF = [('fold', 0), ('fold', 0), ('fold', 0), ('fold', 0), ('raise', 1.5)]  # Assuming UTG, MP, CO, and BTN have folded, SB has raised

    positions_to_solve = [
        UTG_OPEN, 
        MP_OPEN, 
        CO_OPEN,  
        BTN_OPEN, 
        SB_OPEN, 
        MP_DEF,  
        CO_UTG_DEF,
        CO_MP_DEF, 
        BTN_UTG_DEF, 
        BTN_MP_DEF, 
        BTN_CO_DEF, 
        SB_UTG_DEF, 
        SB_MP_DEF, 
        SB_CO_DEF,
        SB_BTN_DEF,
        BB_DEF,
        BB_MP_DEF,
        BB_CO_DEF,
        BB_BTN_DEF,
        BB_SB_DEF,
    ]

    position_names = [
        "UTG_OPEN", 
        "MP_OPEN", 
        "CO_OPEN",  
        "BTN_OPEN", 
        "SB_OPEN", 
        "MP_DEF",  
        "CO_UTG_DEF",
        "CO_MP_DEF", 
        "BTN_UTG_DEF", 
        "BTN_MP_DEF", 
        "BTN_CO_DEF", 
        "SB_UTG_DEF", 
        "SB_MP_DEF", 
        "SB_CO_DEF",
        "SB_BTN_DEF",
        "BB_DEF",
        "BB_MP_DEF",
        "BB_CO_DEF",
        "BB_BTN_DEF",
        "BB_SB_DEF",
    ]

    positions_dict = {str(pos): name for pos, name in zip(positions_to_solve, position_names)}

    return positions_to_solve, positions_dict


def main():
    num_players = 6
    num_ai_players = 5

    # pot relative bet-sizings for preflop, flop, turn, and river
    # bet_sizing = [(1.5, 5), (.33, .70), (.40, .82, 1.2), (.75, 1.2, 2)]
    
    # bet_sizing = [(1.5, ), (), (), ()]

    bet_sizing = [(1.5, 2.0), (), (), ()]

    # bet_sizing = [(1.5, ), (.33, .70), (.40, .82, 1.2), (.75, 1.2, 2)]

    # bet_sizing = [(1.5, ), (.33, .70), (.40, .82, 1.2), (.75, 1.2, 2)]

    positions_to_solve, positions_dict = _6_max_opening()

    # set the deck (if you use a restricted deck, the evaluation will incorrectly evaluate against a full deck)
    SUITS = ['C', 'D', 'H', 'S']
    VALUES = [ '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    # 100 big blinds
    initial_chips = 1000
    small_blind = 5
    big_blind = 10

    # Pretty sure this is deprecated.. just leave it at 1.
    num_showdown_simulations = 1

    # how many times iterations over positions to solve should we perform?
    num_smoothing_iterations = 1

    # **Number of iterations to run the CFR algorithm**
    num_cfr_iterations = 10000
    realtime_cfr_iterations = 200
    cfr_depth = 6
    cfr_realtime_depth = 6
    
    # Depth at which to start Monte Carlo Simulation.
    monte_carlo_depth = 9999

    # Depth at which to start pruning regret and strategy sums.
    prune_depth = 9999
    # Chance-probability at which to start declaring a node "terminal"
    prune_probability = 1e-6

    # Train the AI player using the CFR algorithm
    cfr_trainer = CFRTrainer(num_cfr_iterations, realtime_cfr_iterations, num_showdown_simulations, cfr_depth, cfr_realtime_depth, num_players, initial_chips, small_blind, big_blind, bet_sizing, SUITS, VALUES, monte_carlo_depth, prune_depth, prune_probability)
    strategy_list = cfr_trainer.train(positions_to_solve * num_smoothing_iterations)
    plot_hands(strategy_list, SUITS, VALUES, positions_dict)


    # num_hands = 5
    # game = PokerGame(num_players, initial_chips, num_ai_players, small_blind, big_blind, bet_sizing, cfr_trainer, SUITS, VALUES)
    # # Play the game
    # game.play_game(num_hands)
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
    # Be sure your '../results/' directory has > 70G
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

    z = np.memmap('../results/river_f.npy', mode = 'c', dtype = np.float32, shape = (int(comb(n, 2)) * (int(comb(n, k)) - dupes), 1))

    # #########################################################################################################
    # #########################################################################################################

    # we will also need to modify a kmeans clustering algorithm to use memmapped objects to avoid memory prob


    # highly recommended.
    precompute = True

    if precompute:

        t = time.time()

        print('precomputing river centers')
        centers = ccluster.kmc2(z, rvr_clstrs, chain_length=50, afkmc2=True)
        np.save('../results/cntrs_RIVER', centers)
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

    np.save('../results/adjcntrs_RIVER', miniK.cluster_centers_)
    np.save('../results/lbls_RIVER', miniK.labels_)


    adjcntrs = np.load('../results/adjcntrs.npy', mmap_mode = 'r')
    lbls = np.load('../results/lbls.npy', mmap_mode = 'r')

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



    turn_prob_dist = np.memmap('../results/prob_dist_TURN.npy', mode = 'c', dtype = np.float32, shape = (int(comb(n, 2)) * ((int(comb(n, 4)) - dupes)), n - k - 2))
    print(turn_prob_dist)


    # #########################################################################################################
    # #########################################################################################################


    if precompute:
        t = time.time()
        
        print('precomputing turn centers')
        centers_TURN = ccluster.kmc2(turn_prob_dist, trn_clstrs, chain_length=50, afkmc2=True)
        np.save('../results/cntrs_TURN', centers_TURN)

        print('Time spent precalculating centers: ' + str((time.time() - t)/60) + 'm')
    else:
        centers_TURN = None


    # #########################################################################################################
    # #########################################################################################################

    t = time.time()
    print('computing turn clusters')
    miniK = MiniBatchKMeans(n_clusters = trn_clstrs, batch_size=(trn_clstrs//2)*(threads * 265), tol = 10e-8, max_no_improvement = None, init = centers_TURN, verbose=False, n_init=1).fit(turn_prob_dist)
    print('Time spent: ' + str((time.time() - t)/60) + 'm')


    np.save('../results/adjcntrs_TURN', miniK.cluster_centers_)
    np.save('../results/lbls_TURN', miniK.labels_)

    adjcntrs = np.load('../results/adjcntrs_TURN.npy', mmap_mode = 'r')
    lbls = np.load('../results/lbls_TURN.npy', mmap_mode = 'r')

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


    flop_prob_dist = np.memmap('../results/prob_dist_FLOP.npy', mode = 'c', dtype = np.float32, shape = (int(comb(n, 2)) * ((int(comb(n, 3)) - dupes)), n - k - 1))

    print(flop_prob_dist)
    #########################################################################################################
    #########################################################################################################


    #########################################################################################################


    if precompute:
        t = time.time()
        
        print('precomputing centers ---- FLOP')
        centers_FLOP = ccluster.kmc2(flop_prob_dist, flp_clstrs, chain_length=50, afkmc2=True)
        np.save('../results/cntrs_FLOP', centers_FLOP)

        print('Time spent precalculating centers: ' + str((time.time() - t)/60) + 'm')
    else:
        centers_FLOP = None

    t = time.time()
    print('computing flop clusters')
    miniK = MiniBatchKMeans(n_clusters = flp_clstrs, batch_size=(flp_clstrs//2)*(threads * 265), tol = 10e-8, max_no_improvement = None, init = centers_FLOP, verbose=False, n_init=1).fit(flop_prob_dist)
    print('Time spent: ' + str((time.time() - t)/60) + 'm')


    np.save('../results/adjcntrs_FLOP', miniK.cluster_centers_)
    np.save('../results/lbls_FLOP', miniK.labels_)


    adjcntrs = np.load('../results/adjcntrs_FLOP.npy', mmap_mode = 'r')
    lbls = np.load('../results/lbls_FLOP.npy', mmap_mode = 'r')

    #print(adjcntrs)

    #########################################################################################################
    # print('Saving Flop')
    # pd.DataFrame(flop_prob_dist[:int(comb(n, k-2)) - dupes]).to_csv('flop_prob_dist_23s(24cards).csv')
    # pd.DataFrame(list(map(lambda x: adjcntrs[x], lbls[:int(comb(n, k-2)) - dupes]))).to_csv('flop_prob_dist_clst_23s.csv')
    # print("saved")
    #########################################################################################################


def hand_position(hand, ranks):
    ranks = list(reversed(ranks))
    rank1, rank2 = hand[0], hand[1]
    suited = hand[2] == 's'
    offsuit = hand[2] == 'o'
    pair = rank1 == rank2

    i = ranks.index(rank1)
    j = ranks.index(rank2)

    if pair:
        return i, j
    elif suited:
        return (i, j)
    elif offsuit:
        return (j, i)
    else:
        raise ValueError("Invalid hand format")

def plot_hands(strategy_list, suits=None, ranks=None, positions_dict={}):
    strategy_df = pd.DataFrame(strategy_list)
    strategy_df.columns = ['Position', 'Betting History', 'Hand', 'Strategy']
    strategy_df['Betting History'] = strategy_df['Betting History'].apply(lambda x: str(x))
    strategy_df.to_csv('../results/strategy.csv', index=False)
    strategy_df = strategy_df[strategy_df['Strategy'] != {}]
    strategy_df.reset_index(inplace=True, drop=True)

    # Get unique combinations of 'Position' and 'Betting History'
    unique_combinations = strategy_df[['Position', 'Betting History']].drop_duplicates()

    # Iterate over each unique combination
    for _, row in unique_combinations.iterrows():
        position = row['Position']
        betting_history = row['Betting History']

        # Filter the DataFrame based on the current unique combination
        position_df = strategy_df[(strategy_df['Position'] == position) & (strategy_df['Betting History'] == betting_history)].reset_index(drop=True)

        strategy_proportions = pd.DataFrame()
        for index, row in position_df.iterrows():
            for action, value in row.Strategy.items():
                action_type = f"{action[0]}_{action[1]}"
                if action_type not in strategy_proportions.columns:
                    strategy_proportions[action_type] = 0
                strategy_proportions.at[index, action_type] = value

        strategy_proportions.fillna(0, inplace=True)
        strategy_proportions.reset_index(drop=True, inplace=True)

        subplot_size = np.array([.5, .5])
        ncols = len(ranks)
        nrows = len(ranks)
        figsize = subplot_size * [ncols, nrows]

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()

        # Define colors for specific actions dynamically
        unique_actions = strategy_proportions.columns
        color_map = {}

        # Generate gradient colors for raises
        raise_colors = plt.cm.Reds(np.linspace(0.3, 1, 100))  # Gradient from light red to dark red

        for action in unique_actions:
            if action.startswith('fold'):
                color_map[action] = 'lightblue'
            elif action.startswith('call'):
                color_map[action] = 'green'
            elif action.startswith('raise'):
                raise_size = float(action.split('_')[1])
                color_map[action] = raise_colors[int(raise_size * 50) % 100]  # Scale and mod to stay within bounds
            elif action.startswith('all-in'):
                color_map[action] = 'darkred'

        for idx, row in strategy_proportions.iterrows():
            hand = position_df.loc[idx, 'Hand']
            try:
                pos = hand_position(hand, ranks)
            except ValueError:
                continue

            ax = axes[pos[0] * ncols + pos[1]]
            ax.set_title(hand, size=5, pad=-500)
            ax.set_ylim([0, 1])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # enforce color ordering in plots
            row_items = sorted(row.items(), key=lambda x: (x[0].startswith('all-in'), x[0].startswith('raise'), x[0].startswith('call'), x[0].startswith('fold')))

            bottom = 0
            for action, value in row_items:
                if value > 0:
                    ax.bar(0, value, bottom=bottom, color=color_map[action])
                    bottom += value

        total_positions = len(strategy_proportions)
        if total_positions % ncols != 0:
            for j in range(total_positions, ncols * nrows):
                fig.delaxes(axes[j])

        # Create legend with dynamic colors
        patches = [mpatches.Patch(color=color_map[action], label=action) for action in unique_actions]
        plt.legend(handles=patches, loc='upper left')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.suptitle(f'Strategy for Position: {positions_dict.get(betting_history, betting_history)}', fontsize=13, x=.95, y=.9, rotation=-90)        
        plt.savefig(f'../results/charts/{positions_dict.get(betting_history, betting_history)}_Range.png')
        plt.close()
    
if __name__ == "__main__":
    # cluster()
    main()