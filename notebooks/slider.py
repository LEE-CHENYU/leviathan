# import sys 
# sys.path.append("..")

from matplotlib.widgets import Slider, Button

import matplotlib.pyplot as plt

from Leviathan.Island import Island, Member

import time

round_min = 0
round_max = 204
round_step = 1

island_dict = {}
for round in range(round_min, round_max, round_step):
    island = Island.load_from_pickle(f"/Users/Harry/Documents/Programs/Leviathan/data/Dec/11_23-19/{round}.pkl")
    island_dict[round] = island

fig, axs = plt.subplots(1, 2, figsize=(10, 7))
island.land.plot(axs, show_id=False)

fig.subplots_adjust(bottom=0.25)

ax_round = fig.add_axes([0.25, 0.1, 0.65, 0.03])
round_slider = Slider(
    ax=ax_round,
    label='round',
    valmin=round_min,
    valmax=round_max,
    valinit=round_min,
    valstep=round_step
)

def update(val):
    round = int(round_slider.val / round_step) * round_step
    # island = Island.load_from_pickle(f"data/Dec/07_22-10/{round}.pkl")
    island_dict[round].land.plot(axs, show_id=False)
    fig.canvas.draw_idle()

round_slider.on_changed(update)


# animate_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
# button = Button(animate_ax, 'Animate', hovercolor='0.975')

# def animate(event):
#     for round in range(round_min, round_max, round_step):
#         island_dict[round].land.plot(axs)
#         fig.canvas.draw_idle()
#         time.sleep(10000)
    

# button.on_clicked(animate)



plt.show()