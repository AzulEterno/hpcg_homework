import sys
import json
from collections import defaultdict
import traceback


def nested_dict():
    """Helper function to create nested dictionaries"""
    return defaultdict(nested_dict)

def dictify(d):
    """Convert defaultdict to regular dict recursively"""
    if isinstance(d, defaultdict):
        d = {k: dictify(v) for k, v in d.items()}
    return d

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def resolve_keys_with_value(keys:list[str],value:str,previous_dict = {}):
    if len(keys) <= 1:
        if is_int(value):
            value = int(value)
        elif is_float(value):
            value = float(value)
        previous_dict[keys[0]] = value
        
    else:
        previous_dict[keys[0]] = resolve_keys_with_value(keys=keys[1:],value=value,previous_dict=previous_dict.get(keys[0],{}))
    return previous_dict



def main():
    data = {}

    for line in sys.stdin:
        line = line.strip()
        if not line or line.count("=") <= 0 or (line.count("##########=") > 0):
            continue
        
        try:
            # Split the line into key and value
            key, value = line.split('=') 
            
            key_parts = key.split('::')
            
            if (value ==""):
                continue
            
            data.update(resolve_keys_with_value(key_parts,value=value,previous_dict=data))



            
        except:
            traceback.print_exc()
            print(f"Error line :\"{line}\"")


    

    # Print the JSON representation
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()

