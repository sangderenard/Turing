import numpy as np
from . import DoubleBuffer


import random
import time


def monte_carlo_torture(trials=10, steps=300):
    print(f"Starting Monte Carlo DoubleBuffer torture test: {trials} trials, {steps} steps each...")
    total_errors = 0
    for trial in range(trials):
        width = random.randint(8, 256)
        height = random.randint(4, 128)
        roll_length = random.randint(1, 16)
        num_agents = random.randint(1, 8)
        db = DoubleBuffer(roll_length=roll_length, num_agents=num_agents)
        errors = 0
        for frame in range(steps):
            agent = random.randint(0, num_agents-1)
            arr = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            db.write_frame(arr, agent_idx=agent)
            # Randomly decide to read
            if random.random() < 0.7:
                out = db.read_frame(agent_idx=agent)
                if out is not None:
                    if out.shape != (height, width) or out.dtype != np.uint8:
                        print(f"[ERROR] Trial {trial} Frame {frame} shape/dtype mismatch agent {agent}")
                        errors += 1
        print(f"Trial {trial+1}/{trials}: width={width}, height={height}, roll_length={roll_length}, num_agents={num_agents}, Errors: {errors}")
        total_errors += errors
    print(f"Monte Carlo torture test complete. Total errors: {total_errors}")

if __name__ == "__main__":
    monte_carlo_torture(trials=10, steps=300)
