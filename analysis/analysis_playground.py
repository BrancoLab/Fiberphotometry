# %%
import os 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt


from fcutils.maths.filtering import *
from fcutils.maths.stimuli_detection import *
from fcutils.plotting.colors import *

mvmtc = darkturquoise
ldrc = salmon
blueled = lightskyblue
violetled = violet
ldr_onc = orange
ldr_offc = darkorange


# %%
main_fld = "F:\\"
exp_flds = ["240120_id_994832"]

b, v, mvmt, ldr = [], [], [], []
for f in exp_flds:
    data = pd.read_csv(os.path.join(main_fld, f, "sensors_data.csv")) 
    b.extend(list(data.ch_0_motion.values[2:]))
    v.extend(list(data.ch_0_signal.values[2:]))
    mvmt.extend(list(data.behav_mvmt.values[2:]))
    ldr.extend(list(data.ldr.values[2:]))

b, v, mvmt, ldr = np.array(b), np.array(v), np.array(mvmt), np.array(ldr)
# %%
f, axarr = plt.subplots(nrows=4, figsize=(16, 12))

axarr[0].plot(b, color=blueled)
axarr[1].plot(v, color=violetled)
axarr[2].plot(mvmt, color=mvmtc)
axarr[3].plot(ldr, color=ldrc)
# %%
f, ax = plt.subplots(figsize=(16, 12))
ax.scatter(v, b, alpha=.3)

regressor = LinearRegression()  
regressor.fit(v.reshape(-1, 1), b.reshape(-1, 1)) #training the algorithm
expected_b = v*regressor.coef_[0][0] + regressor.intercept_[0]
corrected_b = b-expected_b

print("intercept", regressor.intercept_[0]) 
print("slope", regressor.coef_[0])


# %%
# Stimuli detection
ldr2 = np.zeros_like(ldr)
ldr2[ldr>4.45] = 1

ldr_onset = np.where(np.diff(ldr2) > .5)[0]
ldr_offset = np.where(np.diff(ldr2) < -.5)[0]


f, ax = plt.subplots(figsize=(16, 12))


ax.plot(np.diff(ldr2), color=ldrc)
ax.scatter(ldr_onset, [.1 for on in ldr_onset], color=ldrc)
ax.scatter(ldr_offset, [-.1 for on in ldr_offset], color=ldrc)

print(len(ldr_onset), len(ldr_offset))

# %%
# ---------------------------------------------------------------------------- #
#                                  ! MAIN PLOT                                 #
# ---------------------------------------------------------------------------- #



shift = 20

f, axarr = plt.subplots(figsize=(18, 12), nrows=4)

axarr[0].plot(b, color=blueled, zorder=99)
axarr[1].plot(v, color=violetled, zorder=99)

axarr[2].plot(corrected_b, color='k', alpha=.8)
#axarr[2].plot(line_smoother(corrected_b), color='r', lw=2, zorder=99)

axarr[3].plot(ldr2, color=ldrc, lw=2)


for g in ldr_onset:
    for i in [0, 1, 2]:
        axarr[i].axvline(g, ls="--", color=ldr_onc)

for g in ldr_offset:
    for i in [0, 1, 2]:
        axarr[i].axvline(g, ls="--", color=ldr_offc)


# %%
# Traces for LED on
preframes, postframes = 80, 100
stimdur=26
f, ax = plt.subplots(figsize=(16, 5), sharex=True, )

on, off = [], [], 

for i,l in enumerate(ldr_onset):
    # axarr[0].plot(corrected_b[l-preframes:l+postframes], color=cyan, alpha=.3)
    on.append(corrected_b[l-preframes:l+postframes])


for i,l in enumerate(ldr_offset):
    # axarr[1].plot(corrected_b[l-preframes:l+postframes], color=red, alpha=.3)
    off.append(corrected_b[l-preframes:l+postframes])
    
ax.plot(np.mean(np.vstack(on), axis=0), color=seagreen, lw=4, zorder=99)

ax.fill_between(np.arange(len(on[0])), np.mean(np.vstack(on), axis=0)-stats.sem(np.vstack(on), axis=0), np.mean(np.vstack(on), axis=0)+stats.sem(np.vstack(on), axis=0), 
                        color='k', alpha=.3)

# axarr[2].plot(np.mean(np.vstack(both), axis=0), color='k', lw=3)
# axarr[3].plot(np.mean(np.vstack(both_off), axis=0), color='k', lw=3)


ax.axvline(preframes, color=salmon, lw=4, ls="--", alpha=.6)
ax.axvline(preframes+stimdur, color=salmon, lw=4, ls="--", alpha=.6)


X = np.arange(0, 180, 16)
_ = ax.set(title='GRATING ALIGNED', ylabel='Normalised fluorescence', xlabel="time(s)", ylim=[-2, 4],
                xticks=X, xticklabels=((X-80)/16).astype(np.int8))

# f.tight_layout()
# 



# %%
plt.show()

# %%

# %%
