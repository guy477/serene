from setuptools import setup
from Cython.Build import cythonize

import numpy
# cimport numpy
# python ccluster-helper.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("poker.game_state", ["poker/game_state.pyx"]),
    Extension("poker.poker_game", ["poker/poker_game.pyx"]),
    Extension("poker.player", ["poker/player.pyx"]),
    Extension("poker.ai_player", ["poker/ai_player.pyx"]),
    Extension("poker.cfr", ["poker/cfr.pyx"]),
]

setup(
    name="poker",
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[numpy.get_include()]
)