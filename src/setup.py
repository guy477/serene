from setuptools import setup
from Cython.Build import cythonize

import numpy
# cimport numpy
# python ccluster-helper.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("poker._utils.eval.eval", ["poker/_utils/eval/eval.pyx"]),
    Extension("poker._utils.ccluster.ccluster", ["poker/_utils/ccluster/ccluster.pyx"]),
    Extension("poker._utils._utils", ["poker/_utils/_utils.pyx"]),

    Extension("poker.core.hash_table", ["poker/core/hash_table.pyx"]),
    Extension("poker.core.local_manager", ["poker/core/local_manager.pyx"]),
    
    Extension("poker.game.deck", ["poker/game/deck.pyx"]),
    Extension("poker.game.game_state", ["poker/game/game_state.pyx"]),
    Extension("poker.game.player", ["poker/game/player.pyx"]),
    Extension("poker.game.poker_game", ["poker/game/poker_game.pyx"]),
    
    Extension("poker.cfr.cfr", ["poker/cfr/cfr.pyx"]),
]

setup(
    name="poker",
    ext_modules=cythonize(extensions, language_level=3, annotate=False),
    include_dirs=[numpy.get_include()]
)