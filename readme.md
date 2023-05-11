

This repository contains a barebones implementation of Texas Hold'em written in Cython 0.29.28 (Python 3.8.8).


The functionality defined is to be integrated with a CFR algorithm written in Cython. This CFR implementation will eventually be linked with an abstraction algorithm I've previously worked on.


End goal is to create an efficient (and scalable) implementation of poker that serves a dual purpose of being a GTO solver and a practice environment for a learning player. 


To install please follow the below steps:

- 1. Install Anaconda Package Manager for Python.

- 2. Create the Anaconda Enviornment:
```
conda create --name Serene python=3.8.8 numpy pandas cython
```
- 3. Activate the Anaconda Environment
```
conda activate Serene
```
- 4. Navigate your command line to the project directory and go into the source folder.
```
cd Your/Path/To/serene/src
```
- 5. Compile the project:
```
python setup.py build_ext --inplace
```
- 6. Run the current implementation
```
python main.py
```