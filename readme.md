
---

# Texas Hold'em Solvers

This repository contains a long-running personal project of mine that explores Texas Hold'em solvers using my personal 'research' build of CFR written in Cython 0.29.28 (Python 3.8.8) for my Macbook Pro M1.

- **Note:** The code is poorly written and maintained.
- **Note:** The code is not optimized for interpretability.

## Things It Does (sort of)
- Abstraction (Preflop) (Postflop\* abstraction is not current)
- Blueprint Training (solve positions iteratively to build a solution index)\*
- CFR Minimization:
    - **Monte Carlo sampling**
        - Starting at a depth - switch to monte carlo based sampling.
    - **Pruning**
        - **By Depth**
            - Prune Regret and Strategy sums if they're accumulated below a specified depth each iteration.
        - **By Probability**: 
            - Skip to a terminal states if the current player's likelihood to arrive there is below a specified threshold.
    - **Custom Hashing**
        - I add a "prune" flag to each value; purging at the end of each iteration accordingly
            - A custom hashing object allows this to be implemented seamlessly.
            - It's boring, but i think it's clever!!
        - Easier Abstractions and Key Management \*(TODO: Migrate gamestate abstraction to utilities)
    - **Customizable Bet Sizing**
        - Define what bet sizings to use on each street.
    - **Any player count**
        - Heads-up to 9-max \*(I've only tested heads up and 6-max...)
    - **Abstraction** \*
        - Fully abstracted preflop
        - Postflop is WIP. \*(See [ccluster.pyx](src/poker/ccluster.pyx) for that nightmare)
    - **Double Precision**
        - You'd expect nothing less... But.. really..
            - I just wanted a place to talk about how you run out of compute precision after a certain depth because of layered probability spaces and what not.
                - You could use some clever bit shifting stuff; but is it worth the effort? 
                    - I bet there's research on this..
                        - (5 minutes later...) It's probably buried in some paper.. back to code.

```* = WIP```

## Things to do (maybe)
- Correct progress_to_showdown in GameState
- Solve each GameState backwards (solve the blinds, working backwards to the UTG)
- Better abstract GameState representations
- More to come...

## Example
The following is a range chart that displays how often the (UTG) should open raise 1.5x the pot (1.5bb * 1.5 = 2.25bb) in 6-Max Texas Hold'em. Iterations Search depth and gamestate hashing are heavily restricted due to memory limitations and performance constraints. Regardless, the sampled range below reflects a loose version of what you might see on GTO Wizard.

![Preflop range for current commit](./dat/EX%20Preflop%20UTG%20Open%206%20Max.png)

## Installation
To install the current implementation, please follow the steps below:

**Note:** The current implementation was built on a Macbook Pro M1.

1. **Install Anaconda Package Manager for Python.**

2. **Create the Anaconda Environment:**
    ```sh
    conda create --name Serene python=3.8.8 numpy pandas cython scikit-learn tqdm matplotlib psutil
    ```

3. **Activate the Anaconda Environment:**
    ```sh
    conda activate Serene
    ```

4. **Navigate your command line to the project directory and go into the source folder:**
    ```sh
    cd Your/Path/To/serene/src
    ```

5. **Compile the project:**
    ```sh
    python setup.py build_ext --inplace
    ```

6. **Run the current implementation:**
    ```sh
    python main.py
    ```

---