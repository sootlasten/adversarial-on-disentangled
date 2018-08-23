"""Parses the info file an adversarial attack generates and outputs
   valid latex table rows."""
import re
from collections import defaultdict
import numpy as np


def find_nth_character(str1, substr, n):
    pos = -1
    for x in range(n):
        pos = str1.find(substr, pos+1)
        if pos == -1:
            return None
    return pos

AS_targets_dict = defaultdict(list)
AS_ignore_targets_dict = defaultdict(list)

filename = "/home/stensootla/projects/adversarial-on-disentangled/ent-wdec-adv-inf.txt"
with open(filename, 'r') as f:
  for line in f.readlines():
    line = line.strip()
    cur_target = int(line[2])
    
    c_target, c_ignore_target = re.findall(r'\(\d+/\d+\)', line)

    vals = list(map(lambda x: int(x), c_target[1:-1].split('/')))
    AS_target = round(vals[0] / vals[1] * 100, 2)
    
    vals = list(map(lambda x: int(x), c_ignore_target[1:-1].split('/')))
    AS_ignore_target = round(vals[0] / vals[1] * 100, 2)
    
    AS_targets_dict[cur_target].append(AS_target)
    AS_ignore_targets_dict[cur_target].append(AS_ignore_target)
    
print("AS_ignore_target")
row = ""
full_AS_ignore_target = []
full_AS_target = []
for digit_class in range(10):
  AS_ignore_target_marg = round(np.mean(AS_ignore_targets_dict[digit_class]), 2)
  AS_target_marg = round(np.mean(AS_targets_dict[digit_class]), 2)

  full_AS_ignore_target.extend(AS_ignore_targets_dict[digit_class])
  full_AS_target.extend(AS_targets_dict[digit_class])

  row += " & \makecell{{{:.2f}\% \\\\ ({:.2f}\%)}}".format(
    AS_ignore_target_marg, AS_target_marg)
  
print(round(np.mean(full_AS_ignore_target), 2), 
  round(np.mean(full_AS_target), 2))
#print(row)

