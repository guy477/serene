This repository contains a long-running personal project of mine that explores Texas Hold'em solvers using my personal 'research' build of CFR written in Cython 0.29.28 (Python 3.8.8). 

- **The code is poorly written and maintained** 

- **The code is not optimized for interpretability.**

## Example
The folling is a range chart the displays how ofte the (SB) should open raise 1.5x the pot (1.5bb*1.5 = 2.25bb) in heads up Texas Hold'em. Search depth and gamestate hashing is heavily restricted due to memory limitations and performance constraints. Regardless, the sampled range below 'makes sense'.
![Preflop range for current commit](./HU%20SB%20PREFLOP%20EXAMPLE.png)


# Install
To install the current implementation please follow the below steps:

**The current implementation was built on a Macbook Pro M1**

- 1. Install Anaconda Package Manager for Python.

- 2. Create the Anaconda Enviornment:
```
conda create --name Serene python=3.8.8 numpy pandas cython scikit-learn tqdm matplotlib
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