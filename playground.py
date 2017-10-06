
import random
import numpy as np
import pandas as pd

print random.randint(-1, 1)


LOOK_BACK = 10

#df = pd.read_csv('/home/bam/Documents/PyCharmsProjects/RLTrading/csv/convertedFull.csv', index_col=0, parse_dates=['Date'], usecols=[0, 1, 7, 8, 9])
#nextState = df.iloc[1 : 1 + LOOK_BACK]

#converted = np.array(nextState, dtype=pd.Series)[:,1:4]



#a = np.array(nextState.tail(1)['Date'].dt.hour)[0]


pp = [1, 2, 3, 4, 5]

np.random.shuffle(pp)


abc = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


wbd = [1, 4, 2 ,8, 3, 7 ,9, 20, -1, 2, 3, 4, 6]


sub = abc[0:15]

print sub


print ((9 in abc) is not True)


print list(set([1, 2, 3]) & set([2, 3, 4]))


abc[0:0] = [-11]

print abc[-1]


print sorted(wbd), "qqq"

import os

duration = 3  # second
freq = 440  # Hz
os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

