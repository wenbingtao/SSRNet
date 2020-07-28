import numpy as np


class dtu_mvs_params:
	def __init__(self):
		self.class_freq = np.asarray([50.0, 50.0])
		self.class_weights = -np.log(self.class_freq / 100.0)
		self.num_classes = len(self.class_freq) + 1
		self.color_map = [[255, 255, 255], # 0 (white)
						  [128, 0, 0],  # 1 (red)
			 			  [255, 225, 25]]   # 2 (yellow)