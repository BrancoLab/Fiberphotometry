# %%
import os 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline


from utils.maths.filtering import *
from utils.maths.stimuli_detection import *
from utils.colors import *

left_led_color = darkturquoise
right_led_color = orange
both_leds_color = salmon
both_leds_off_color = darkseagreen

blueled = lightskyblue
violetled = violet



# %%
main_fld = "Z:\\swc\\branco\\rig_photometry\\191203"
exp_fld = "191203_CE972_4"

data = pd.read_csv(os.path.join(main_fld, exp_fld, "sensors_data.csv"))
b, v = data.ch_0_motion.values[2:], data.ch_0_signal.values[2:]
left, right = data.left_led_on.values[2:], data.right_led_on.values[2:]
# %%
f, axarr = plt.subplots(nrows=2)

axarr[0].plot(v)
axarr[1].plot(b)

# %%
f, ax = plt.subplots()
ax.scatter(v, b, alpha=.3)

regressor = LinearRegression()  
regressor.fit(v.reshape(-1, 1), b.reshape(-1, 1)) #training the algorithm


print("intercept", regressor.intercept_[0]) 
print("slope", regressor.coef_[0])

# %%

# ---------------------------------------------------------------------------- #
#                                  ! MAIN PLOT                                 #
# ---------------------------------------------------------------------------- #

expected_b = v*regressor.coef_[0][0] + regressor.intercept_[0]
corrected_b = b-expected_b

left_stim_onsets = np.where(np.diff(left) > .5)[0]
right_stim_onsets = np.where(np.diff(right) > .5)[0]
left_stim_offsets = np.where(np.diff(left) < -.5)[0]
right_stim_offsets = np.where(np.diff(right) < -.5)[0]

f, axarr = plt.subplots(figsize=(18, 4), nrows=4)

axarr[0].plot(b, color=blueled, zorder=99)
axarr[1].plot(v, color=violetled, zorder=99)

axarr[2].plot(corrected_b, color='k', alpha=.8)
axarr[2].plot(line_smoother(corrected_b), color='r', lw=2, zorder=99)

axarr[3].plot(left, color=left_led_color, lw=2)
axarr[3].plot(right, color=right_led_color, lw=2)

for n, g in enumerate(left_stim_onsets):
    for i in [0, 1, 2]:
        if n%2 == 0:
            axarr[i].axvline(g, ls="--", color=left_led_color)
        else:
            axarr[i].axvline(g, ls="--", color=both_leds_color)

for g in right_stim_onsets:
    for i in [0, 1, 2]:
        axarr[i].axvline(g, ls="--", color=right_led_color)

for g in right_stim_offsets:
    for i in [0, 1, 2]:
        axarr[i].axvline(g, ls="--", color=both_leds_off_color)


axarr[3].scatter(left_stim_onsets, [1.1 for i in left_stim_onsets], color=left_led_color, s=50)
axarr[3].scatter(right_stim_onsets, [1.2 for i in right_stim_onsets], color=right_led_color, s=50)
axarr[3].scatter(left_stim_offsets, [-.1 for i in left_stim_onsets], color=left_led_color, s=50)
axarr[3].scatter(right_stim_offsets, [-.2 for i in right_stim_onsets], color=right_led_color, s=50)


# %%
# AVG per LED on
both_off = np.where((left == 0) & (right == 0))
left_on = np.where(left == 1)
right_on = np.where(right == 1)
both_on = np.where((left == 1) & (right == 1))

f, ax = plt.subplots()
for i, idx in enumerate([both_off, left_on, right_on, both_on]):
    ax.errorbar(i, np.mean(corrected_b[idx]), fmt='o', yerr=stats.sem(corrected_b[idx]), color='b')
_ = ax.set(xticks=np.arange(4), xticklabels=[ 'none', 'left', 'right', 'both',])

# %%
# Traces for LED on
preframes, postframes = 40, 40
f, axarr = plt.subplots(nrows=4, figsize=(10, 10))

ll, rr, both, both_off = [], [], [], []

for i,l in enumerate(left_stim_onsets):
    if i % 2 == 0:
        axarr[0].plot(corrected_b[l-preframes:l+postframes], color=cyan)
        ll.append(corrected_b[l-preframes:l+postframes])
    else:
        axarr[2].plot(corrected_b[l-preframes:l+postframes], color=orange)
        both.append(corrected_b[l-preframes:l+postframes])

for i, r in enumerate(right_stim_onsets):
    axarr[1].plot(corrected_b[r-preframes:r+postframes], color=darkturquoise)
    rr.append(corrected_b[r-preframes:r+postframes])


for i, o in enumerate(right_stim_offsets):
    axarr[3].plot(corrected_b[o-preframes:o+postframes], color=darkred)
    both_off.append(corrected_b[o-preframes:o+postframes])

axarr[0].plot(np.mean(np.vstack(ll), axis=0), color='k', lw=3)
axarr[1].plot(np.mean(np.vstack(rr), axis=0), color='k', lw=3)
axarr[2].plot(np.mean(np.vstack(both), axis=0), color='k', lw=3)
axarr[3].plot(np.mean(np.vstack(both_off), axis=0), color='k', lw=3)

for ax in list(axarr):
    ax.axvline(preframes, color='k')

_ =axarr[0].set(title='Left LED onset aligned')
_ =axarr[1].set(title='Right LED onset aligned')
_ =axarr[2].set(title='Both LED onset aligned')
_ =axarr[3].set(title='Both LED offset aligned')

f.tight_layout()




# %%
plt.show()

# %%

# %%
