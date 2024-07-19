import os
from setuptools import setup
from Cython.Build import cythonize

import numpy
# cimport numpy
# python ccluster-helper.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize


from setuptools.command.build_ext import build_ext
import shutil


## NOTE: This will move all .c, .h, and .so files from the source directory to the build directory
class CustomBuildExt(build_ext):
    def build_extension(self, ext):
        # Call the original build_extension method
        super().build_extension(ext)

        # Move .c and .h files to the build directory
        for source in ext.sources:
            source_dir = os.path.dirname(source)
            build_dir = os.path.join('build', source_dir)
            os.makedirs(build_dir, exist_ok=True)

            # skip eval's arrays.h file for version control
            if 'arrays.h' in source:
                continue

            if source.endswith('.c') or source.endswith('.h') or source.endswith('.cpp'):
                shutil.move(source, os.path.join(build_dir, os.path.basename(source)))

        # Move .so files to the build directory
        so_filename = self.get_ext_filename(ext.name)
        so_file = os.path.join(self.build_lib, so_filename)
        so_build_dir = os.path.join('build', os.path.dirname(so_filename))
        os.makedirs(so_build_dir, exist_ok=True)
        shutil.move(so_file, os.path.join(so_build_dir, os.path.basename(so_file)))

    def get_outputs(self):
        outputs = super().get_outputs()
        # Update the output paths to reflect the moved .so files
        new_outputs = []
        for output in outputs:
            if output.endswith('.so'):
                build_output = os.path.join('build', output)
                new_outputs.append(build_output)
            else:
                new_outputs.append(output)
        return new_outputs

## Define the exntensions to compile
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

build_dir = "build"

## Compile files to the build directory using CustomBuildExt to segregate file types
setup(
    name="poker",
    ext_modules=cythonize(extensions, language_level=3, annotate=False),
    include_dirs=[numpy.get_include()],
    cmdclass={'build_ext': CustomBuildExt},
    options={
        'build_ext': {
            'build_lib': 'build',
            'build_temp': os.path.join('build', 'temp'),
        }
    },
    script_args=['build_ext', '--build-lib', 'build', '--build-temp', os.path.join('build', 'temp')]
)

# Copy the source files (pyx/pxd) to the build directory (.c, .h, .o, .so)
def copy_tree(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)

src_dir = os.path.dirname(os.path.abspath(__file__))
dst_dir = os.path.join(src_dir, build_dir)

copy_tree(src_dir + '/poker', dst_dir + '/poker')


## NOTE: I'm sure there's a better way to do this