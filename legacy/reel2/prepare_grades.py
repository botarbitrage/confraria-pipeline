"""Grade the full x4 master once per look, so the video camera can move freely."""
import numpy as np, time
from PIL import Image
import build_shots as B
M = Image.open('ampoule_x4.png').convert('RGB')
geom = (0.0, 0.0, 768.0, 1376.0)   # whole frame, source coords
for key, shot in (('base', 't1_sala'), ('t3', 't3_preco'), ('t4', 't4_pergunta')):
    t0 = time.time()
    g = B.grade(shot, M, geom)
    np.save(f'grade_{key}.npy', np.asarray(g))
    print(key, g.size, round(time.time() - t0, 1), 's', flush=True)
