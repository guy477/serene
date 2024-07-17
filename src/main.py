
from poker.poker_game import PokerGame
from poker.cfr import CFRTrainer
import poker.ccluster as ccluster
from poker._utils import LocalManager, _6_max_opening, _6_max_simple_postflop

from multiprocessing import Manager
import pickle

import ast
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.special import comb
from sklearn.cluster import MiniBatchKMeans





def train():
    num_players = 6

    # pot relative bet-sizings for preflop, flop, turn, and river
    # bet_sizing = [(1.5, 5), (.33, .70), (.40, .82, 1.2), (.75, 1.2, 2)]
    
    # bet_sizing = [(1.5, ), (), (), ()]

    bet_sizing = [(1.5, 2.0,), (.33, 1,), (), ()]

    # bet_sizing = [(1.5, ), (.33, .70), (.40, .82, 1.2), (.75, 1.2, 2)]

    # bet_sizing = [(1.5, ), (.33, .70), (.40, .82, 1.2), (.75, 1.2, 2)]

    positions_to_solve, positions_dict = _6_max_opening()

    # Train Only SB v BB:
    positions_to_solve = positions_to_solve[2:3]

    # set the deck (if you use a restricted deck, the evaluation will incorrectly evaluate against a full deck)
    SUITS = ['C', 'D', 'H', 'S']
    VALUES = [ '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    # 100 big blinds
    initial_chips = 1000
    small_blind = 5
    big_blind = 10

    # Pretty sure this is deprecated.. just leave it at 1.
    num_showdown_simulations = 1

    # Specify the number of times to iteratate over `positions_to_solve`.
    ## Fun Fact: This is one way to construct a blueprint strategy.
    num_smoothing_iterations = 1

    # **Number of iterations to run the CFR algorithm**
    num_cfr_iterations = 500
    cfr_depth = 3
    
    # Depth at which to start Monte Carlo Simulation.
    monte_carlo_depth = 9999

    # Depth at which to start pruning regret and strategy sums.
    prune_depth = 4
    # Chance-probability at which to start declaring a node "terminal"
    prune_probability = 1e-8

    # Train the AI player using the CFR algorithm
    # local_manager = LocalManager('dat/pickles/regret_sum_S1_3k_D11_P10.pkl', 'dat/pickles/strategy_sum_S1_3k_D11_P10.pkl')
    local_manager = LocalManager('dat/_tmp/_regret_sum.pkl', 'dat/_tmp/_strategy_sum.pkl')
    
    cfr_trainer = CFRTrainer(num_cfr_iterations, num_showdown_simulations, cfr_depth, num_players, initial_chips, small_blind, big_blind, bet_sizing, SUITS, VALUES, monte_carlo_depth, prune_depth, prune_probability)
    strategy_list, _local_manager = cfr_trainer.train(local_manager, positions_to_solve * num_smoothing_iterations, save_pickle = True) # TODO add pickle paths to regret/strat solves.
    plot_hands(strategy_list, SUITS, VALUES, positions_dict)

    local_manager.save('dat/pickles/regret_sum_SB_PROBFIX.pkl', 'dat/pickles/strategy_sum_SB_PROBFIX.pkl')


def play():
    num_players = 6
    num_ai_players = 0

    # pot relative bet-sizings for preflop, flop, turn, and river
    bet_sizing = [(1.5, 2.0,), (.5, 1,), (.40, .82, 1.2,), (.75, 1.2, 2,)]

    # set the deck (if you use a restricted deck, the evaluation will incorrectly evaluate against a full deck)
    SUITS = ['C', 'D', 'H', 'S']
    VALUES = [ '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    # 100 big blinds
    initial_chips = 1000
    small_blind = 5
    big_blind = 10

    # **Number of iterations to run the CFR algorithm**
    num_cfr_iterations = 500
    cfr_depth = 2
    
    # Depth at which to start Monte Carlo Simulation.
    monte_carlo_depth = 9999

    # Depth at which to start pruning regret and strategy sums.
    prune_depth = 2
    # Chance-probability at which to start declaring a node "terminal"
    prune_probability = 1e-8

    
    # The cfr_trainer will handle blueprint strategy management. Strategies are saved to disk, so we can just define a new CFR trainer
    # For an AI player who will play in realtime.
    cfr_trainer = CFRTrainer(num_cfr_iterations, 1, cfr_depth, num_players, initial_chips, small_blind, big_blind, bet_sizing, SUITS, VALUES, monte_carlo_depth, prune_depth, prune_probability)

    local_manager = LocalManager('dat/_tmp/_regret_sum.pkl', 'dat/_tmp/_strategy_sum.pkl')
    num_hands = 100
    game = PokerGame(num_players, initial_chips, num_ai_players, small_blind, big_blind, bet_sizing, cfr_trainer, local_manager, SUITS, VALUES)
    # # Play the game
    game.play_game(num_hands)
    print('\n\n')

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

def plot_hands(strategy_list=None, suits=None, ranks=None, positions_dict={}):
    if strategy_list:
        strategy_df = pd.DataFrame(strategy_list)
        strategy_df.columns = ['Position', 'Betting History', 'Hand', 'Strategy']
        strategy_df['Betting History'] = strategy_df['Betting History'].apply(lambda x: str(x))
        strategy_df.to_csv('../results/strategy.csv', index=False)
    else:
        strategy_df = pd.read_csv('../results/strategy.csv')
        strategy_df['Strategy'] = strategy_df['Strategy'].apply(ast.literal_eval)

    strategy_df.drop_duplicates(subset = ['Position', 'Betting History', 'Hand'], keep = 'last', inplace=True)
    strategy_df = strategy_df[strategy_df['Strategy'] != {}]
    strategy_df.reset_index(inplace=True, drop=True)

    print(strategy_df)

    # Get unique combinations of 'Position' and 'Betting History'
    unique_combinations = strategy_df[['Position', 'Betting History']].drop_duplicates()

    # Iterate over each unique combination
    for _, row in unique_combinations.iterrows():
        position = row['Position']
        action_space = row['Betting History']

        # Filter the DataFrame based on the current unique combination
        position_df = strategy_df[(strategy_df['Position'] == position) & (strategy_df['Betting History'] == action_space)].reset_index(drop=True)

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
        plt.suptitle(f'Strategy for Position: {positions_dict.get(action_space, action_space)}', fontsize=13, x=.95, y=.9, rotation=-90)        
        plt.savefig(f'../results/charts/{positions_dict.get(action_space, action_space)}_Range.png')
        plt.close()
    
if __name__ == "__main__":
    # cluster()
    train()
    # play()

    # local_manager = LocalManager('dat/_tmp/_regret_sum.pkl', 'dat/_tmp/_strategy_sum.pkl')

    # print(local_manager.get_strategy_sum().table)