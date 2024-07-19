import sys
import os
# Add the build directory to the system path
sys.path.insert(0, os.path.abspath('build'))

from poker.game.poker_game import PokerGame
from poker.cfr.cfr import CFRTrainer
import poker._utils.ccluster as ccluster
from poker._utils._utils import _6_max_opening, _2_max_opening, _6_max_simple_postflop
from poker.core.local_manager import LocalManager


from multiprocessing import Manager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.special import comb
from sklearn.cluster import MiniBatchKMeans





def train():
##########
    # set the deck (if you use a restricted deck, the evaluation will assumed a full deck)
    SUITS = ['C', 'D', 'H', 'S']
    VALUES = [ '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    # 100 big blinds
    initial_chips = 1000
    small_blind = 5
    big_blind = 10

    # pot relative bet-sizings for preflop, flop, turn, and river
    bet_sizing = [(1.5, 2.0,), (.33, 1,), (), ()]

    # How many players to solve for
    num_players = 2

    # Fetch action space of preflop positions
    positions_to_solve, positions_dict = _2_max_opening()

    # All heads-up preflop positions to a depth of three-bet defense
    positions_to_solve = positions_to_solve # [:3] # Train Only SB v BB  # [:8] # Train SB v BB; SB v BTN; BB v BTN 

    # Specify the number of times to iteratate over `positions_to_solve`
    num_smoothing_iterations = 1
    positions_to_solve = positions_to_solve * num_smoothing_iterations

##########

    # **Number of iterations to run the CFR algorithm**
    num_cfr_iterations = 1000

    # Actions by folded or all-in players dont count toward depth
    # -> Choose depth to have action end on starting player
    cfr_depth = 1
    
    # Depth at which to start Monte Carlo Simulation.
    monte_carlo_depth = 9999

    # Depth at which to start pruning regret and strategy sums
    prune_depth = 9999
    # Chance-probability at which to start declaring a node "terminal"
    prune_probability = 1e-8

##########

    # Create a training environment and train the model using CFR
    cfr_trainer = CFRTrainer(num_cfr_iterations, cfr_depth, num_players, initial_chips, small_blind, big_blind, bet_sizing, SUITS, VALUES, monte_carlo_depth, prune_depth, prune_probability)
    
    for i, fast_forward_actions in enumerate(positions_to_solve):
        current_position = positions_dict[str(fast_forward_actions)]

        # Define environment directory for current position
        base_path = f'../results/{num_players}/{cfr_depth}/{current_position}'
        pkl_path = base_path + '/pickles/'

        # Load current blueprint strategy
        local_manager = LocalManager(pkl_path)
        
        ## train our hand matrix on the given fast_forward_action space
        local_manager = cfr_trainer.train(local_manager, [fast_forward_actions], save_pickle = True)
        
        ## leverage local_manager to source strategy_list
        strategy_list = cfr_trainer.get_average_strategy_dump(fast_forward_actions, local_manager)
        
        ## Plot the results
        plot_hands(current_position, strategy_list, SUITS, VALUES, base_path)
        
        if i + 1 < len(positions_to_solve):
            ## Copy current local_manager to next position
            next_base_path = f'../results/{num_players}/{cfr_depth}/{positions_dict[str(positions_to_solve[i + 1])]}'
            next_pkl_path = next_base_path + '/pickles/'
            local_manager.base_path = next_pkl_path
            
            ## Save current pkls to new directory.
            local_manager.save()


def play():
    num_players = 2
    num_ai_players = 2

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
    num_cfr_iterations = 5000

    # Leave this at 1 if your blueprint is not fully solved to the depth specified
    #   (i.e. the 6-player solve is missing 4-bets, re-raises by more than 1 player, etc.)
    cfr_depth = 3
    
    # Depth at which to start Monte Carlo Simulation
    monte_carlo_depth = 9999

    # Depth at which to start pruning regret and strategy sums
    prune_depth = 9999
    # Chance-probability at which to start declaring a node "terminal"
    prune_probability = 1e-8

    
    # The cfr_trainer will handle blueprint strategy management. Strategies are saved to disk, so we can just define a new CFR trainer
    cfr_trainer = CFRTrainer(num_cfr_iterations, cfr_depth, num_players, initial_chips, small_blind, big_blind, bet_sizing, SUITS, VALUES, monte_carlo_depth, prune_depth, prune_probability)

    # The earliest positions solved have the broadest node coverage (to support the later positions)
    local_manager = LocalManager('../results/2/4/BB_SB_4B_DEF/pickles/')
    num_hands = 10
    game = PokerGame(num_players, initial_chips, num_ai_players, small_blind, big_blind, bet_sizing, cfr_trainer, local_manager, SUITS, VALUES)
    
    # Play the game
    game.play_game(num_hands)
    print('\n\n')

#########################

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

def plot_hands(position_name, strategy_list, suits=None, ranks=None, base_path = ''):

    strategy_df = pd.DataFrame(strategy_list)
    strategy_df.columns = ['Position', 'Betting History', 'Hand', 'Strategy', 'Reach Probability']
    strategy_df['Betting History'] = strategy_df['Betting History'].apply(lambda x: str(x))
    
    os.makedirs(f'{base_path}/csv_extracts/', exist_ok=True)
    os.makedirs(f'{base_path}/charts/', exist_ok=True)

    strategy_df.to_csv(f'{base_path}/csv_extracts/{position_name}_RANGE.csv', index=False)
    

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
            reach_probability = position_df.loc[idx, 'Reach Probability']
            opacity = 0.6 + 0.4 * reach_probability if reach_probability > 0.2 else 0.05 + 0.55 * reach_probability

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
                    ax.bar(0, value, bottom=bottom, color=color_map[action], alpha=min(opacity, 1))
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
        plt.suptitle(f'Strategy for Position: {position_name}', fontsize=13, x=.95, y=.9, rotation=-90)        
        plt.savefig(f'{base_path}/charts/{position_name}_Range.png')
        plt.close()
    
if __name__ == "__main__":
    # cluster()
    # train()
    play()