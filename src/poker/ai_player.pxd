from .player cimport Player
from .cfr cimport CFRTrainer
from .game_state cimport GameState, display_game_state

cdef class AIPlayer(Player):

    cdef CFRTrainer strategy_trainer
    cdef public dict regret
    cdef public dict strategy_sum

    cpdef get_strategy(self, list available_actions, float[:] probs, GameState game_state)
    cdef initialize_regret_strategy(self)
    cpdef clone(self)