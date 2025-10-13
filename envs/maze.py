import numpy as np


# Configurazione originale per compatibilità
MEDIUM_MAZE_RANDOM_INIT_FIXED_GOAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 'r', 'r', 1, 1, 'r', 'r', 1],
                                      [1, 'r', 'r', 1, 'r', 'r', 'r', 1],
                                      [1, 1, 'r', 'r', 'r', 1, 1, 1],
                                      [1, 'r', 'r', 1, 'r', 'r', 'r', 1],
                                      [1, 'r', 1, 'r', 'r', 1, 'r', 1],
                                      [1, 'r', 'r', 'r', 1, 'g', 'r', 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1]]

# Configurazione originale per compatibilità
MEDIUM_MAZE_FIXED_INIT_RANDOM_GOAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 'r', 'g', 1, 1, 'g', 'g', 1],
                                    [1, 'g', 'g', 1, 'g', 'g', 'g', 1],
                                    [1, 1, 'g', 'g', 'g', 1, 1, 1],
                                    [1, 'g', 'g', 1, 'g', 'g', 'g', 1],
                                    [1, 'g', 1, 'g', 'g', 1, 'g', 1],
                                    [1, 'g', 'g', 'g', 1, 'g', 'g', 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1]]

# Configurazione originale per compatibilità
MEDIUM_MAZE_FIXED_INIT_FIXED_GOAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 'r', 0, 1, 1, 0, 0, 1],
                                    [1, 0, 0, 1, 0, 0, 0, 1],
                                    [1, 1, 0, 0, 0, 1, 1, 1],
                                    [1, 0, 0, 1, 0, 0, 0, 1],
                                    [1, 0, 1, 0, 0, 1, 0, 1],
                                    [1, 0, 0, 0, 1, 0, 'g', 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1]]