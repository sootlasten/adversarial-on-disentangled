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

filename = "/home/stensootla/projects/adversarial-on-disentangled/entangled-weightdecay.txt"
with open(filename, 'r') as f:
  for line in f.readlines():
    line = line.strip()
    cur_source = int(line[0])
    
    c_target, c_ignore_target = re.findall(r'\(\d+/\d+\)', line)

    vals = list(map(lambda x: int(x), c_target[1:-1].split('/')))
    AS_target = round(vals[0] / vals[1] * 100, 2)
    
    vals = list(map(lambda x: int(x), c_ignore_target[1:-1].split('/')))
    AS_ignore_target = round(vals[0] / vals[1] * 100, 2)
    
    AS_targets_dict[cur_source].append(AS_target)
    AS_ignore_targets_dict[cur_source].append(AS_ignore_target)
    
print("AS_ignore_target")
s = ""
for digit_class, AS_ignore_target_list in AS_ignore_targets_dict.items():
  s += " & {:.2f}\%".format(round(np.mean(AS_ignore_target_list), 2))
print(s)

print("\nAS_target")
s = ""
for digit_class, AS_target_list in AS_targets_dict.items():
  s += " & {:.2f}\%".format(round(np.mean(AS_target_list), 2))
print(s)

  
