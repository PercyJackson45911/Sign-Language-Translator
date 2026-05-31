import sys
import time

from tqdm import tqdm

for _ in tqdm(range(10), file=sys.stdout):
    time.sleep(0.1)
