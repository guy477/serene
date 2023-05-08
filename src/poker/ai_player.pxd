from .player cimport Player
from .cfr cimport CFRTrainer
from .game_state cimport GameState, display_game_state

cdef class AIPlayer(Player):
    cdef CFRTrainer strategy_trainer
    cpdef get_action(self, GameState game_state, int player_index)