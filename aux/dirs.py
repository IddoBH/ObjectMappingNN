import os
import sys

for i in range(100):
    os.mkdir(os.path.join(sys.argv[1], 'train', f'set_{i+1}'))
    os.mkdir(os.path.join(sys.argv[1], 'val', f'set_{i+1}'))

