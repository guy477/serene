from .player cimport Player
from .cfr cimport CFRTrainer
from .game_state cimport GameState, display_game_state

cdef class AIPlayer(Player):

    cdef CFRTrainer strategy_trainer

    cpdef get_strategy(self, list available_actions, float[:] probs, GameState game_state)
    cpdef clone(self)