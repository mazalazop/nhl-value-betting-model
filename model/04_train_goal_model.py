"""Train and evaluate a distinct GOAL 1+ model; see --help for frozen study phases."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from henachel.goal_experiment import main

if __name__ == '__main__':
    main()
