import matplotlib.pyplot as plt
import time
import random
 
from Island import Island, Member

round_min = 10
round_max = 300
round_step = 10

island_dict = {}
for round in range(round_min, round_max, round_step):
    island = Island.load_from_pickle(f"data/Dec/07_22-10/{round}.pkl")
    island_dict[round] = island

fig, axs = plt.subplots(1, 2, figsize=(10, 7))
island.land.plot(axs, show_id=False)
 
plt.show()
 
for round in range(round_min, round_max, round_step):
    island_dict[round].land.plot(axs)

    fig.draw()
    # plt.draw()
    # plt.pause(1e-17)
    time.sleep(0.2)

# add this if you don't want the window to disappear at the end
plt.show()
