from .game_state cimport GameState, hand_to_cards
from .ai_player cimport AIPlayer
from .information_set cimport InformationSet

cdef class CFRTrainer:
    cdef public int iterations
    cdef public int cfr_depth

    cdef public int num_players
    cdef public int initial_chips
    cdef public int small_blind
    cdef public int big_blind
    cdef public dict strategy_profiles
    cdef public dict regret_sum
    cdef public dict strategy_sum

    cpdef train(self)

    cpdef train_realtime(self, GameState game_state, int iterations, int cfr_realtime_depth)

    cdef cfr_traverse(self, GameState game_state, int player, float[:] probs, int depth, int max_depth)

    cpdef str get_best_action(self, GameState game_state, int player_index)
    
    cdef float[:] calculate_utilities(self, GameState game_state, int player)

    cdef get_average_strategy(self, AIPlayer player, GameState game_state)