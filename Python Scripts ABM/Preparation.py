import numpy as np
from scipy.stats import skewnorm
from mcerp import correlate, uv
import pickle

empathy = uv(skewnorm(0.7719369, loc=0.06258684, scale=0.3560445))
print('first distribution done')
ambigtol = uv(skewnorm(1.01172724, loc=-0.2153221, scale=0.3222545))
print('second distribution done')
outrage = uv(skewnorm(0.8765333, loc=0.1002369, scale=0.2801581))
print('outrage done')
ojs = uv(skewnorm(0.7366447, loc=0.1589589, scale=0.3849354))
print('ojs done')
vjs = uv(skewnorm(0.7747939, loc=0.1615498, scale=0.3907201))
print('vjs done')
soccomp = uv(skewnorm(0.8856228, loc=0.2920341, scale=0.2747496))
print('done creating the distributions')

c_target = np.array([[1.00000000, -0.08838773, 0.26305828, 0.48623732, 0.29962615, 0.31590676],
                     [-0.08838773, 1.00000000, -0.24084587, -0.12450355, -0.33068515, 0.21549226],
                     [0.26305828, -0.24084587, 1.00000000, 0.34378025, 0.44516420, -0.01179118],
                     [0.48623732, -0.12450355, 0.34378025, 1.00000000, 0.51285403, 0.09620488],
                     [0.29962615, -0.33068515, 0.44516420, 0.51285403, 1.00000000, -0.09147703],
                     [0.31590676, 0.21549226, -0.01179118, 0.09620488, -0.09147703, 1.00000000]])

correlate([empathy, ambigtol, outrage, ojs, vjs, soccomp], c_target)
print('done with the correlation')

with open('empathy.pkl', 'wb') as out:
    pickle.dump(empathy, out, pickle.HIGHEST_PROTOCOL)
with open('ambigtol.pkl', 'wb') as out:
    pickle.dump(ambigtol, out, pickle.HIGHEST_PROTOCOL)
with open('outrage.pkl', 'wb') as out:
    pickle.dump(outrage, out, pickle.HIGHEST_PROTOCOL)
with open('ojs.pkl', 'wb') as out:
    pickle.dump(ojs, out, pickle.HIGHEST_PROTOCOL)
with open('vjs.pkl', 'wb') as out:
    pickle.dump(vjs, out, pickle.HIGHEST_PROTOCOL)
with open('soccomp.pkl', 'wb') as out:
    pickle.dump(soccomp, out, pickle.HIGHEST_PROTOCOL)
