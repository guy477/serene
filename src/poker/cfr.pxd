from .game_state cimport GameState

cdef class CFRTrainer:
    cdef public int iterations

    cpdef train(self)

    cpdef double traverse_game_tree(self, GameState game_state, int player_index, double probability)

    cpdef str get_best_action(self, GameState game_state, int player_index)

    #cpdef double cfr(self, GameState game_state, int player_index, double probability)