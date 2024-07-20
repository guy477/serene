
import numpy as np
cimport numpy as np
import random
import psutil
import gc, time
import itertools


from collections import defaultdict
from tqdm import tqdm


from multiprocessing import Pool, Manager, set_start_method


# No more shared memory >:(
set_start_method('spawn', force=True)

cdef class CFRTrainer:

    def __init__(self, int iterations, int cfr_depth, int num_players, int initial_chips, int small_blind, int big_blind, list bet_sizing, list suits=SUITS, list values=VALUES, int prune_depth = 9999, double prune_probability = 1e-8, local_manager=None):
        self.iterations = iterations
        self.suits = suits
        self.values = values

        self.num_players = num_players
        self.initial_chips = initial_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.bet_sizing = bet_sizing


        self.cfr = CFR(cfr_depth, prune_depth, prune_probability)
    
    cpdef train(self, local_manager, list positions_to_solve = [], list hands = [], bint save_pickle=False):
        FUNCTION_START_TIME = time.time()
        
        if not hands:
            hands = self.generate_hands()

        # Convert string hands to unsigned long long hands.
        hands = [(card_str_to_int(hand[0]), card_str_to_int(hand[1])) for hand in hands]        

        # Shuffle the list with a specific seed for reproducibility
        rng = random.Random(42)  # Use any seed value you prefer
        rng.shuffle(hands)

        for i, fast_forward_action in enumerate(positions_to_solve):
            
            local_manager = self.parallel_train(hands, fast_forward_action, local_manager, save_pickle)

            if save_pickle:
                local_manager.save()  

        print(f'Time taken: {time.time() - FUNCTION_START_TIME}')

        return local_manager


    def parallel_train(self, hands, fast_forward_actions, local_manager, save_pickle):#(psutil.cpu_count(logical=True) -1) * 2

        ### Managed Objects
        manager = Manager()
        regret_global_accumulator = manager.dict()
        strategy_global_accumulator = manager.dict()
        calculated = manager.dict()

        def process_batch(batch_hands):                                                                        # NOTE: Since we _spawn_ each pooled process, local_manager gets copied to each child process. If the blueprint gets too large, we make local manager read only and global; then only update a local copy of the explorable nodes.
            hands = [(hand, regret_global_accumulator, strategy_global_accumulator, calculated, fast_forward_actions, local_manager) for hand in batch_hands]
            with Pool(processes = psutil.cpu_count(logical=True) - 1) as pool: #psutil.cpu_count(logical=True) - 1
                try:
                    pool.starmap(self.process_hand_wrapper, hands)
                except Exception as e:
                    pool.terminate()
                    raise
                finally:
                    pool.close()
                    pool.join()

        batch_size = (psutil.cpu_count(logical=True) - 1)

        num_batches = (len(hands) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches)):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, len(hands))
            batch_hands = hands[start_index:end_index]

            process_batch(batch_hands)

            # Migrate local solutions to global solutions
            local_manager.get_regret_sum().update(dict(regret_global_accumulator))
            local_manager.get_strategy_sum().update(dict(strategy_global_accumulator))
            
            if save_pickle:
                local_manager.save()  

            #############
            regret_complexity_calculation = (len(regret_global_accumulator))/len(batch_hands)
            strategy_complexity_calculation = (len(strategy_global_accumulator))/len(batch_hands)
            print(f'Update Complexity: {regret_complexity_calculation if self.cfr.cfr_depth > 1 else 1}')
            print(f'Update Complexity: {strategy_complexity_calculation if self.cfr.cfr_depth > 1 else 1}')
            #############

            # Clear local solutions for next run
            regret_global_accumulator.clear()
            strategy_global_accumulator.clear()
            gc.collect()

        return local_manager


    def process_hand_wrapper(self, hand, regret_global_accumulator, strategy_global_accumulator, calculated, fast_forward_actions, local_manager):
        self.process_hand(hand, regret_global_accumulator, strategy_global_accumulator, calculated, fast_forward_actions, local_manager)


    def process_hand(self, hand, regret_global_accumulator, strategy_global_accumulator, calculated, fast_forward_actions, local_manager):
        
        

        cdef GameState game_state = GameState([Player(self.initial_chips, self.bet_sizing, False) for _ in range(self.num_players)], self.small_blind, self.big_blind, True, self.suits, self.values) 

        ffw_probs = self.fast_forward_gamestate(hand, game_state, fast_forward_actions, local_manager)
        if ffw_probs == 'ERROR':
            print('Provided hand is incompatible with Fast Forward Actions')
            return

        #############
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        print(f'TRAINING: ({game_state.get_current_player().position}) --- {abstract_hand(hand[0], hand[1])}')
        print(fast_forward_actions)
        print(f'Regret Complexity: {len(local_manager.get_regret_sum())}')
        print(f'Strategy Complexity: {len(local_manager.get_strategy_sum())}')
        # print()
        #############

        player = game_state.get_current_player()
        strategy = self.cfr.get_average_strategy(player, game_state, local_manager)
        player_hash = player.hash(game_state)

        #############
        print(player_hash)
        print(f"Init Regret Sum: {local_manager.get_regret_sum().get(player_hash, defaultdict(self.default_double))}")
        print(f"Init Strategy Sum: {local_manager.get_strategy_sum().get(player_hash, defaultdict(self.default_double))}")
        print(f"Init Average Strategy For Hand {abstract_hand(hand[0], hand[1])}: {strategy}")
        #############

        total_fails = 0
        for iter_num in tqdm(range(self.iterations)):

            ffw_probs = self.fast_forward_gamestate(hand, game_state, fast_forward_actions, local_manager)
            
            if ffw_probs[game_state.player_index] < self.cfr.prune_probability_threshold:
                break # pretty sure the ffw_probs is deterministic.
                continue

            self.cfr.cfr_traverse(game_state.clone(), ffw_probs, 0, local_manager)
        


        player = game_state.get_current_player()
        player_hash = player.hash(game_state)


        #############
        ## NOTE: Output traversal strategy for current node.
        strategy = self.cfr.get_strategy(player.get_available_actions(game_state), ffw_probs, game_state, player, local_manager)
        regret = local_manager.get_regret_sum()[player_hash]
        print(f"Node Reach Probability: {list(ffw_probs)[game_state.player_index]}")
        print(player_hash)

        print(f"Post Strategy For Hand {abstract_hand(hand[0], hand[1])}: {strategy}")
        print(f"Post Regret Sums For Hand {abstract_hand(hand[0], hand[1])}: {regret}")
        #############
        ## NOTE: Average strategy is for the user
        strategy = self.cfr.get_average_strategy(player, game_state, local_manager)
        print(f"Post Average Strategy For Hand {abstract_hand(hand[0], hand[1])}: {strategy}")
        #############

        ### Prune local regret and strategy sums to save memory at the global level.
        local_manager.get_regret_sum().prune()
        local_manager.get_strategy_sum().prune()
    
        ### Merge Local Results with Global Accumulator
        dynamic_merge_dicts(local_manager.get_regret_sum(), regret_global_accumulator)
        dynamic_merge_dicts(local_manager.get_strategy_sum(), strategy_global_accumulator)


    cdef fast_forward_gamestate(self, object hand, GameState game_state, list fast_forward_actions, LocalManager local_manager, int attempts = 0):
        cdef dict strategy
        cdef list ignore_cards
        cdef GameState cloned_game_state
        cdef double[:] new_probs
        cdef double total
        cdef object action
        cdef Player current_player
        
        # Accumulate cards that need to be handled specially
        ignore_cards = [action[1] for action in fast_forward_actions if action[0] == 'PUBLIC']
        ignore_cards.extend(hand)

        if len(set(ignore_cards)) != len(ignore_cards):
            return 'ERROR'

        # Set up preflop - not dealing any card that appears in the list of ignore_cards
        game_state.setup_preflop(ignore_cards)

        # Create game_state representation to set player hand
        cloned_game_state = game_state.clone()
        for action in fast_forward_actions:
            if action[0] != 'PUBLIC':
                cloned_game_state.step(action)

        # Save the current player index
        player_index_tmp = game_state.player_index

        # Update game_state with the cloned game state
        game_state.player_index = cloned_game_state.player_index

        # Update player hand with the provided hand
        game_state.update_current_hand(hand)

        # Restore the original player index
        game_state.player_index = player_index_tmp

        # Set probability space for CFR traversal
        new_probs = np.ones(self.num_players, dtype=np.float64)

        for action in fast_forward_actions:
            if action[0] == 'PUBLIC':
                # Add custom card to the board and update action space
                game_state.board |= action[1]
                game_state.action_space[game_state.cur_round_index].insert(0, ('PUBLIC', action))
                continue

            # Get the CFR strategy
            current_player = game_state.get_current_player()
            strategy = self.cfr.get_strategy(current_player.get_available_actions(game_state), new_probs, game_state, current_player, local_manager)

            # Update probabilities
            new_probs[game_state.player_index] *= strategy[action]

            if game_state.step(action) and game_state.cur_round_index < 4:
                # Handle the end of the round
                ignore_cards = [new_action for new_action in game_state.action_space[game_state.cur_round_index] if new_action[0] == 'PUBLIC']

                for new_action in ignore_cards:
                    game_state.board -= new_action[1][1]
                    game_state.deck.add(new_action[1][1])
                    game_state.action_space[game_state.cur_round_index].remove(new_action)

                # Normalize probabilities
                total = np.sum(new_probs)
                if total > 0:
                    for i in range(self.num_players):
                        new_probs[i] /= total
                else:
                    for i in range(self.num_players):
                        new_probs[i] = 1

        return new_probs

    cpdef get_average_strategy_dump(self, fast_forward_actions, local_manager):
        cdef GameState game_state = GameState([Player(self.initial_chips, self.bet_sizing, False) for _ in range(self.num_players)], self.small_blind, self.big_blind, True, self.suits, self.values)
        aggregate_hand_dump = []
        hands = self.generate_hands()
        hands = [(card_str_to_int(hand[0]), card_str_to_int(hand[1])) for hand in hands]
        for hand in tqdm(hands):

            ffw_probs = self.fast_forward_gamestate(hand, game_state, fast_forward_actions, local_manager)
            
            player = game_state.get_current_player()
            strategy = self.get_average_strategy(player, game_state, local_manager)
            cur_hand_dump = (player.position, fast_forward_actions, abstract_hand(hand[0], hand[1]), strategy, ffw_probs[game_state.player_index])
            aggregate_hand_dump.append(cur_hand_dump)

        return aggregate_hand_dump




    cpdef generate_hands(self):
        cards = [r + s for r in reversed(self.values) for s in self.suits]

        hands = list(sorted(itertools.combinations(cards, 2)))
        
        return hands

    cpdef double default_double(self):
        ### Idk why this is necessary. 
        ### creating "float"s in cython seem to have half precision.
        ###     could be wrong. 
        return 0.0
