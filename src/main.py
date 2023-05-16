
from poker.poker_game import PokerGame
from poker.cfr import CFRTrainer
import poker.ccluster as ccluster

import time
import numpy as np
import pandas as pd

from scipy.special import comb
from sklearn.cluster import MiniBatchKMeans

def main():
    num_players = 6
    num_ai_players = 6

    # pot relative bet-sizings for preflop, flop, turn, and river
    bet_sizing = [(1, 2), (.33, .70), (.40, .82, 1.2), (.75, 1.2, 2)]

    initial_chips = 1000
    small_blind = 5
    big_blind = 10

    num_iterations = 100
    realtime_iterations = 500
    cfr_depth = 8
    cfr_realtime_depth = 4
    

    # Train the AI player using the CFR algorithm
    cfr_trainer = CFRTrainer(num_iterations, realtime_iterations, cfr_depth, cfr_realtime_depth, num_players, initial_chips, small_blind, big_blind, bet_sizing)
    cfr_trainer.train()

    game = PokerGame(num_players, initial_chips, num_ai_players, small_blind, big_blind, bet_sizing, cfr_trainer)

    # Play the game
    # game.play_game(num_hands=1)
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

if __name__ == "__main__":
    # cluster()
    main()