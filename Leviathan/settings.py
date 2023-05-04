import os
import numpy as np

# Get the path of the current file
_setting_file_path = os.path.abspath(__file__)
package_path = os.path.dirname(_setting_file_path)
name_list = np.loadtxt(package_path + "/name_list.txt", dtype=str)