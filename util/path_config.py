import sys

open3d_path = '/mnt/A/jokery/projects/Open3D_test3/src/build/lib/'
tc_path = '/mnt/A/jokery/projects/08_2/'

sys.path.append(open3d_path)
from py3d import *
def get_tc_path():
	return tc_path
