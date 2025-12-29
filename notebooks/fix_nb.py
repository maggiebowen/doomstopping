import json
import os

path = '/Users/maggiebowen/Documents/GitHub/doomstopping/notebooks/02_model_inference_test.ipynb'

print(f"Reading {path}...")
with open(path, 'r') as f:
    nb = json.load(f)

count = 0
for i, cell in enumerate(nb['cells']):
    if 'source' in cell:
        new_source = []
        for line in cell['source']:
            # Check for strings ending with backslash+n (which is \\n in python string literal for regex, 
            # but here we deal with the loaded string value).
            # If the JSON was "line\\n", python string is 'line\n' (len 2 at end).
            if line.endswith('\\n'):
                # Replace the last 2 chars '\n' with a real newline '\n'
                fixed_line = line[:-2] + '\n'
                new_source.append(fixed_line)
                count += 1
            else:
                new_source.append(line)
        cell['source'] = new_source

print(f"Fixed {count} lines.")

with open(path, 'w') as f:
    json.dump(nb, f, indent=4)
print("Saved notebook.")
