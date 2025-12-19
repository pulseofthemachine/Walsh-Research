"""
Configurator. 
Reads config files and command line arguments to update global variables.
"""
import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's a config file
        print(f"Overriding config with {arg}")
        with open(arg) as f:
            exec(f.read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                val = literal_eval(val)
            except:
                pass
            globals()[key] = val
            print(f"Overriding: {key} = {val}")
        else:
            raise ValueError(f"Unknown config key: {key}")