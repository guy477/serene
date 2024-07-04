
---

# Texas Hold'em Solvers

This repository contains a long-running personal project of mine that explores Texas Hold'em solvers using my personal 'research' build of CFR written in Cython 0.29.28 (Python 3.8.8).

- **Note:** The code is poorly written and maintained.
- **Note:** The code is not optimized for interpretability.

## Example
The following is a range chart that displays how often the (UTG) should open raise 1.5x the pot (1.5bb * 1.5 = 2.25bb) in 6-Max Texas Hold'em. Iterations Search depth and gamestate hashing are heavily restricted due to memory limitations and performance constraints. Regardless, the sampled range below 'makes sense' and took about 10 minutes to generate on my Macbook Pro M1.

![Preflop range for current commit](./dat/EX%20Preflop%20UTG%20Open%206%20Max.png)

## Installation
To install the current implementation, please follow the steps below:

**Note:** The current implementation was built on a Macbook Pro M1.

1. **Install Anaconda Package Manager for Python.**

2. **Create the Anaconda Environment:**
    ```sh
    conda create --name Serene python=3.8.8 numpy pandas cython scikit-learn tqdm matplotlib
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