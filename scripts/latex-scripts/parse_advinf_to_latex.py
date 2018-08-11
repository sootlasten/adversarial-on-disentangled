"""Parses the info file an adversarial attack generates and outputs
   valid latex table rows."""
import re

def find_nth_character(str1, substr, n):
    pos = -1
    for x in range(n):
        pos = str1.find(substr, pos+1)
        if pos == -1:
            return None
    return pos


filename = "/home/stensootla/projects/adversarial-on-disentangled/entangled-weightdecay.txt"
sources = []
with open(filename, 'r') as f:
  for line in f.readlines():
    line = line.strip()
    cur_source = int(line[0])
    if cur_source not in sources:
      sources.append(cur_source)    
      row = "\\textbf{{{}}}".format(cur_source)

    c_target, c_ignore_target = re.findall(r'\(\d+/\d+\)', line)

    vals = list(map(lambda x: int(x), c_target[1:-1].split('/')))
    AS_target = round(vals[0] / vals[1] * 100, 2)
    
    vals = list(map(lambda x: int(x), c_ignore_target[1:-1].split('/')))
    AS_ignore_target = round(vals[0] / vals[1] * 100, 2)
    
    row += " & \makecell{{{:.2f}\% \\\\ ({:.2f}\%)}}".format(AS_ignore_target, AS_target)
    if row.count('&') == 9: 
      idx = find_nth_character(row, '&', len(sources))
      if idx is not None:
        row = row[:idx+1] + " - &" + row[idx+1:]
      else:
        row += ' & -'
      row += " \\\\ \\hline"
      print(row)
  
