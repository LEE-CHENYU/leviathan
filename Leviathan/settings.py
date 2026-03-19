import os
import numpy as np

# Get the path of the current file
_setting_file_path = os.path.abspath(__file__)
package_path = os.path.dirname(_setting_file_path)

# Load the name list from the package directory
name_list_path = os.path.join(package_path, "name_list.txt")
if not os.path.exists(name_list_path):
    raise FileNotFoundError(f"name_list.txt not found at {name_list_path}")

name_list = np.loadtxt(name_list_path, dtype=str)
