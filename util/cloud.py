import numpy as np
import os

from joblib import Parallel, delayed
import multiprocessing
import subprocess

import random

from util.path_config import *

def read_txt_summary(file_path):
	with open(file_path) as f:
		labels = f.readlines()
		lb = [(c.strip())[1] for c in labels]
	return np.asarray(lb)

def expand_dim_to_batch2(array, local_batch_size, dummy_val=-1):
	sp = np.shape(array)
	out_arr = np.zeros((local_batch_size, sp[1]))
	out_arr.fill(dummy_val)
	out_arr[0:sp[0], :] = array
	return out_arr

def expand_dim_to_batch1(array, batch_size, dummy_val=0):
	sp = np.shape(array)
	out_arr = np.zeros((batch_size))
	out_arr.fill(dummy_val)
	out_arr[0:sp[0]] = array
	return out_arr

def get_pooling_mask(pooling):
	mask = np.asarray(pooling > 0, dtype='float32')
	sp = np.shape(mask)
	for i in range(0, sp[0]):
		mult = np.count_nonzero(mask[i, :])
		if mult == 0:
			mask[i, :] *= 0
		else:
			mask[i, :] *= 1/mult
	return mask

def invert_index_map(idx_map):
	remapped_idx = {}
	for i in range(0, len(idx_map)):
		remapped_idx[idx_map[i]] = i
	return remapped_idx

def remap_indices(idx_map, cloud_subset):
	sp = np.shape(cloud_subset)
	cloud_subset.flatten()
	out = [idx_map.get(i, -1) for i in cloud_subset.flatten()]
	return np.reshape(np.asarray(out), (sp[0], sp[1]))

class ScanData():

	def __init__(self):
		self.vertex_clouds = []
		self.vertex_trees = []
		self.clouds = []
		self.labels_gt = []
		self.labels_pr = []
		self.ratios_pr = []
		self.pre_pr = []
		self.conv_ind_vertex = []
		self.pool_ind_vertex = []
		self.depth_vertex = []
		self.normal_vertex = []
		self.conv_ind = []
		self.pool_ind = []
		self.depth = []
		self.normal = []
		self.trees = []
		self.partial_vertex = []
		self.batches = []

	def load(self, file_path, par):
		print('file_path',file_path)
		num_scales = par.num_scales
		print('num_scales',num_scales)
		for i in range(0, num_scales):

			# fname = os.path.join(file_path, "scale_" + str(i) + ".npz")
			file_names = {}
			file_names['cloud'] = (os.path.join(file_path, "point_scale_" + str(i) + ".ply"))
			file_names['point_index'] = (os.path.join(file_path, "point_index_scale_" + str(i) + ".bin"))
			file_names['point_dn'] = (os.path.join(file_path, "point_dn_scale_" + str(i) + ".bin"))
			file_names['vertex_cloud'] = (os.path.join(file_path, "vertex_scale_" + str(i) + ".ply"))
			file_names['vertex_index'] = (os.path.join(file_path, "vertex_index_scale_" + str(i) + ".bin"))
			file_names['vertex_dn'] = (os.path.join(file_path, "vertex_dn_scale_" + str(i) + ".bin"))

			if i != 0:
				file_names['point_cubic'] = (os.path.join(file_path, "point_cubic_index_scale_" + str(i) + ".bin"))
				file_names['vertex_cubic'] = (os.path.join(file_path, "vertex_cubic_index_scale_" + str(i) + ".bin"))
			if i == 0:
				file_names['labels_gt'] = (os.path.join(file_path, "vertex_all_label.txt"))
				file_names['partial_label'] = (os.path.join(file_path, "vertex_label.bin"))

			file_all_exists = True
			for names in file_names.values():
				if not os.path.exists(names):
					file_all_exists = False
					continue
			if file_all_exists == False:
				continue

			cloud = read_point_cloud(file_names['cloud'])
			# cloud.points = Vector3dVector(l['points'])
			# cloud.colors = Vector3dVector(l['colors'])
			tree = KDTreeFlann(cloud)

			vertex_cloud = read_point_cloud(file_names['vertex_cloud'])
			# vertex_cloud.points = Vector3dVector(l['vertex'])
			vertex_tree = KDTreeFlann(vertex_cloud)

			patch_size_pow2 = par.filter_size*par.filter_size

			l = {}
			l['nn_conv_ind_points'] = np.fromfile(file_names['point_index'], dtype=np.int32)
			l['nn_conv_ind_points'] = l['nn_conv_ind_points'].reshape((l['nn_conv_ind_points'].shape[0] // patch_size_pow2,
																	   patch_size_pow2))
			l['nn_conv_ind_points'] = l['nn_conv_ind_points'].T
			l['nn_conv_ind_vertex'] = np.fromfile(file_names['vertex_index'], dtype=np.int32)
			l['nn_conv_ind_vertex'] = l['nn_conv_ind_vertex'].reshape((l['nn_conv_ind_vertex'].shape[0] // patch_size_pow2,
																	   patch_size_pow2))
			l['nn_conv_ind_vertex'] = l['nn_conv_ind_vertex'].T

			point_depth_normal = np.fromfile(file_names['point_dn'], dtype=np.float32)
			point_depth_normal = point_depth_normal.reshape(
				(point_depth_normal.shape[0] // (patch_size_pow2*4), patch_size_pow2*4))

			l['depth_points'] = (point_depth_normal.T)[0:patch_size_pow2, :]
			l['normal_points'] = (point_depth_normal.T)[patch_size_pow2:patch_size_pow2*4, :]

			vertex_depth_normal = np.fromfile(file_names['vertex_dn'], dtype=np.float32)
			vertex_depth_normal = vertex_depth_normal.reshape(
				(vertex_depth_normal.shape[0] // (patch_size_pow2*4), patch_size_pow2*4))

			l['depth_vertex'] = (vertex_depth_normal.T)[0:patch_size_pow2, :]
			l['normal_vertex'] = (vertex_depth_normal.T)[patch_size_pow2:patch_size_pow2*4, :]

			if i == 0:
				l['pool_ind_points'] = None
				l['pool_ind_vertex'] = None

				l['labels_gt'] = np.fromfile(file_names['labels_gt'], dtype=np.int32, sep=' ')
				self.labels_gt.append(l['labels_gt'])

				l['partial_vertex'] = np.fromfile(file_names['partial_label'], dtype=np.int32)
				l['partial_vertex'] = l['partial_vertex'].reshape((l['partial_vertex'].shape[0] // 2, 2))
				self.partial_vertex.append(l['partial_vertex'])

				if 'labels_pr' in l.keys():
					self.labels_pr.append(l['labels_pr'])
				else:
					self.labels_pr.append(np.zeros(np.shape(self.labels_gt[0])))
				self.ratios_pr.append(np.zeros(np.shape(self.labels_gt[0])))
				self.pre_pr.append(np.zeros([np.shape(self.labels_gt[0])[0], 2]))

			if i != 0:
				l['pool_ind_points'] = np.fromfile(file_names['point_cubic'], dtype=np.int32)
				l['pool_ind_points'] = l['pool_ind_points'].reshape((l['pool_ind_points'].shape[0] // 8, 8))

				l['pool_ind_vertex'] = np.fromfile(file_names['vertex_cubic'], dtype=np.int32)
				l['pool_ind_vertex'] = l['pool_ind_vertex'].reshape((l['pool_ind_vertex'].shape[0] // 8, 8))

			self.clouds.append(cloud)
			self.trees.append(tree)
			self.vertex_clouds.append(vertex_cloud)
			self.vertex_trees.append(vertex_tree)
			self.conv_ind.append(l['nn_conv_ind_points'])
			self.depth.append(l['depth_points'])
			self.normal.append(l['normal_points'])
			self.pool_ind.append(l['pool_ind_points'])

			self.conv_ind_vertex.append(l['nn_conv_ind_vertex'])
			self.depth_vertex.append(l['depth_vertex'])
			self.normal_vertex.append(l['normal_vertex'])
			self.pool_ind_vertex.append(l['pool_ind_vertex'])

		print("loaded " + file_path.split('/')[-2])

	def save(self, file_path, num_scales=3):
		for i in range(0, num_scales):
			np.savez_compressed(os.path.join(file_path, 'scale_' + str(i) + '.npz'),
					points=np.asarray(self.clouds[i].points),
					colors=np.asarray(self.clouds[i].colors),
					labels_gt=self.labels_gt[i],
					labels_pr=self.labels_pr[i],
					nn_conv_ind=self.conv_ind[i],
					pool_ind=self.pool_ind[i],
					depth=self.depth[i])

	def save_label_pr(self, file_path):
		np.savez_compressed(os.path.join(file_path, 'scale_' + str(i) + '.npz'),
							labels_pr=self.labels_pr[0])

	def remap_depth(self, vmin, vmax):
		num_scales = len(self.depth)
		for i in range(0, num_scales):
			self.depth[i] = np.clip(self.depth[i], vmin, vmax)
			self.depth[i] -= vmin
			self.depth[i] *= 1.0 / (vmax - vmin)

			invalid_index = np.where(np.isfinite(self.depth_vertex[i]) == False)
			(self.depth_vertex[i])[:, (invalid_index[1])] = vmax
			self.depth_vertex[i] = np.clip(self.depth_vertex[i], vmin, vmax)
			self.depth_vertex[i] -= vmin
			self.depth_vertex[i] *= 1.0 / (vmax - vmin)

	def has_points(self, point, radius):
		[k, idx_valid, _] = self.trees[0].search_radius_vector_3d(point, radius=radius)
		return np.count_nonzero(self.labels_gt[0][idx_valid]) > 0

	def has_vertexs(self, point, radius):
		[k, idx_valid, _] = self.vertex_trees[0].search_radius_vector_3d(point, radius=radius)
		return np.count_nonzero(self.labels_gt[0][idx_valid]) > 0

	def has_edges(self, point, radius):
		[k, idx_valid, _] = self.edge_center_trees[0].search_radius_vector_3d(point, radius=radius)
		return np.count_nonzero(self.labels_gt[0][idx_valid]) > 0

	def get_random_valid_point(self, radius=None):
		h = False
		while not h:
			num_points = np.shape(np.asarray(self.clouds[0].points))[0]
			random_ind = random.randint(0, num_points-1)
			random_point = np.asarray(self.clouds[0].points)[random_ind, :]
			if radius is not None:
				h = self.has_points(random_point, radius)
			else:
				h = self.labels_gt[0][random_ind] > 0
		return random_point, random_ind

	def get_random_valid_label2_edge_center(self, radius=None):
		h = False
		while not h:
			# points =
			label2_index = np.where(self.labels_gt[0] > 1)
			# valid_edge_center = self.edge_center_clouds[0].points[label2_index[0]]
			num_points = np.shape(label2_index[0])[0]
			random_ind = random.randint(0, num_points-1)
			random_ind = (label2_index[0])[random_ind]
			random_point = np.asarray(self.edge_center_clouds[0].points)[random_ind, :]
			if radius is not None:
				h = self.has_edges(random_point, radius)
			else:
				h = self.labels_gt[0][random_ind] > 1
		return random_point, random_ind

	def get_random_partial_vertex(self, radius=None):
		h = False
		while not h:
			num_points = np.shape(self.partial_vertex[0])[0]
			random_ind = random.randint(0, num_points-1)
			random_ind = (self.partial_vertex[0])[random_ind, 0]
			random_point = np.asarray(self.vertex_clouds[0].points)[random_ind, :]
			if radius is not None:
				h = self.has_vertexs(random_point, radius)
			else:
				h = self.labels_gt[0][random_ind] > 0
		return random_point, random_ind

	def remap_normals(self, vmin=-1.0, vmax=1.0):
		num_scales = len(self.clouds)
		for i in range(0, num_scales):
			normals = np.asarray(self.clouds[i].normals)
			normals = np.clip(normals, vmin, vmax)
			normals -= vmin
			normals *= 1.0 / (vmax - vmin)
			self.clouds[i].normals = Vector3dVector(normals)

	def get_height(self, scale=0):
		raw_z = np.asarray(self.clouds[scale].points)[:, 2:3]
		max_z = np.max(raw_z)
		min_z = np.min(raw_z)
		return (raw_z - min_z) * 1.0 / (max_z - min_z)

	def assign_labels(self, pr_arr):
		for i in range(0, len(pr_arr)):
			if pr_arr[i] > 0:
				self.labels_pr[0][i] = pr_arr[i]
		self.labels_pr[1] = None
		self.labels_pr[2] = None

	def assign_labels_part(self, pr_arr, idx_map):
		for i in range(0, len(idx_map)):
			if pr_arr[i] > 0:
				self.labels_pr[0][idx_map[i]] = pr_arr[i]
		self.labels_pr[1] = None
		self.labels_pr[2] = None

	def assign_vertex_labels_ratios(self, label_arr, ratio_arr):
		self.labels_pr[0] = label_arr
		self.ratios_pr[0] = ratio_arr[:, label_arr]

	def assign_vertex_labels_ratios_pres(self, label_arr, ratio_arr, pre_arr, length):
		self.labels_pr[0] = label_arr[0:length]
		self.ratios_pr[0] = ratio_arr[0:length, label_arr[0:length]]
		self.pre_pr[0] = pre_arr[0:length, :]

	def assign_edge_labels_part(self, pr_arr, idx_map):
		for i in range(0, idx_map.shape[0]):
			self.labels_pr[0][idx_map[i]] = pr_arr[i] + 1

	def assign_vertex_labels_ratios_part(self, label_arr, ratio_arr, idx_map):
		for i in range(0, idx_map.shape[0]):
			self.labels_pr[0][idx_map[i]] = label_arr[i] + 1
			self.ratios_pr[0][idx_map[i]] = ratio_arr[i][label_arr[i]]

	def assign_vertex_labels_ratios_pres_part(self, label_arr, ratio_arr, pre_arr, idx_map):
		for i in range(0, idx_map.shape[0]):
			self.labels_pr[0][idx_map[i]] = label_arr[i] + 1
			self.ratios_pr[0][idx_map[i]] = ratio_arr[i][label_arr[i]]
			self.pre_pr[0][idx_map[i]] = pre_arr[i, :]

	def extrapolate_labels(self, ref_cloud, ref_labels):
		out_labels = []
		points = np.asarray(ref_cloud.points)
		cnt = 0
		total_lb = 0
		correct_lb = 0

		idx_val = self.labels_pr[0] > 0

		labeled_cloud = PointCloud()
		labeled_cloud.points = Vector3dVector(np.asarray(self.clouds[0].points)[idx_val, :])
		labeled_cloud.colors = Vector3dVector(np.asarray(self.clouds[0].colors)[idx_val, :])
		search_tree = KDTreeFlann(labeled_cloud)
		valid_labels = self.labels_pr[0][idx_val]

		for pt in points:
			gt_lb = ref_labels[cnt]
			if gt_lb > 0:
				[k, idx, _] = search_tree.search_knn_vector_3d(pt, 1)
				pr_lb = valid_labels[idx]
				out_labels.append(int(pr_lb))
				if pr_lb == gt_lb:
					correct_lb += 1
				total_lb += 1
			else:
				out_labels.append(int(0))
			cnt += 1
		accuracy = correct_lb / total_lb
		return out_labels, accuracy


class BatchData():

	def __init__(self):
		self.colors = []
		self.vertex_conv_ind = []
		self.point_conv_ind = []
		self.vertex_pool_ind = []
		self.point_pool_ind = []
		self.vertex_conv_ind = []
		self.vertex_pool_ind = []
		self.labels = []
		self.vertex_depth = []
		self.vertex_normal = []
		self.point_depth = []
		self.point_normal = []
		self.point_index_maps = []
		self.vertex_index_maps = []
		self.edge_index_maps = []
		self.loss_weights = []
		self.edges = []

	def num_points(self):
		return np.shape(self.colors[0])[0]

	def check_valid(self, batch_size):
		return np.shape(self.point_depth[0])[1] <= batch_size and \
			np.shape(self.point_depth[1])[1] <= (batch_size / 2) \
			and np.shape(self.labels[0])[0] <= batch_size


def get_vertex_batch_from_full_scan(full_scan, num_scales, class_weights):
	out = BatchData()

	out.labels.append(np.asarray(full_scan.labels_gt[0]))
	num_vertex = out.labels[0].shape[0]

	curr_w = np.zeros(num_vertex, np.float32)
	for lb_index in range(num_vertex):
		lb = (full_scan.labels_gt[0])[lb_index]
		if lb > 0:
			curr_w[lb_index] = class_weights[int(lb - 1)]
		else:
			curr_w[lb_index] = 0.0
	out.loss_weights.append(curr_w)

	for i in range(0, num_scales):
		out.colors.append(np.asarray(full_scan.clouds[i].colors))

		out.vertex_depth.append(np.asarray(full_scan.depth_vertex[i]))
		out.point_depth.append(np.asarray(full_scan.depth[i]))
		out.vertex_normal.append(np.asarray(full_scan.normal_vertex[i]))
		out.point_normal.append(np.asarray(full_scan.normal[i]))

		out.point_conv_ind.append(np.asarray(full_scan.conv_ind[i].T))
		out.vertex_conv_ind.append(np.asarray(full_scan.conv_ind_vertex[i].T))

		if i > 0:
			pool_ind_vertex = np.asarray(full_scan.pool_ind_vertex[i])
			pool_ind_points = np.asarray(full_scan.pool_ind[i])
		else:
			pool_ind_vertex = None
			pool_ind_points = None
		out.vertex_pool_ind.append(pool_ind_vertex)
		out.point_pool_ind.append(pool_ind_points)
	return out


def get_vertex_scan_part_out(par, point=None, sample_type='POINT'):
	lb_count = 0
	num_scales = len(scan.clouds)
	random_point = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
	while lb_count == 0:
		if point is None:
			if sample_type == 'SPACE':
				min_bound = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
				max_bound = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
				ext = max_bound - min_bound
				random_point = np.random.rand(3) * ext + min_bound
			elif sample_type == 'POINT':
				num_points = np.shape(scan.partial_vertex[0])[0]
				random_ind = random.randint(0, num_points - 1)
				random_ind = (scan.partial_vertex[0])[random_ind, 0]
				random_point = np.asarray(scan.vertex_clouds[0].points)[random_ind, :]
		else:
			random_point = np.asarray(point)

		[k_valid, idx_valid, _] = scan.vertex_trees[0].search_radius_vector_3d(random_point, radius=par.valid_rad)
		idx_valid = np.asarray(idx_valid)
		lbl = np.asarray(scan.labels_gt[0][idx_valid])
		lb_count = np.count_nonzero(lbl)

		if (point is not None) and (lb_count == 0):
			return None

	point_idx_maps = []
	vertex_idx_maps = []
	out = BatchData()

	[k, valid_vertex_idx, _] = scan.vertex_trees[0].search_radius_vector_3d(random_point, radius=par.valid_rad)
	valid_vertex_idx = np.asarray(valid_vertex_idx)
	num_vertex = valid_vertex_idx.shape[0]
	tmp_labels = np.asarray(scan.labels_gt[0][valid_vertex_idx])
	out.labels.append(tmp_labels)

	curr_w = np.zeros(num_vertex, np.float32)
	for lb_index in range(num_vertex):
		lb = tmp_labels[lb_index]
		if lb > 0:
			curr_w[lb_index] = par.d_par.class_weights[int(lb - 1)]
		else:
			curr_w[lb_index] = 0.0
	out.loss_weights.append(curr_w)

	for i in range(0, num_scales):

		if i == 0:
			vertex_idx = valid_vertex_idx
		else:
			[k, vertex_idx, _] = scan.vertex_trees[i].search_radius_vector_3d(random_point, radius=par.full_rf_size())

		vertex_idx = np.asarray(vertex_idx)
		[k, point_idx, _] = scan.trees[i].search_radius_vector_3d(random_point, radius=par.full_rf_size())
		point_idx = np.asarray(point_idx)
		tmp_vertex_num = vertex_idx.shape[0]
		tmp_point_num = point_idx.shape[0]

		out.point_depth.append(np.asarray((scan.depth[i])[:, point_idx]))
		out.vertex_depth.append(np.asarray((scan.depth_vertex[i])[:, vertex_idx]))
		out.point_normal.append(np.asarray((scan.normal[i])[:, point_idx]))
		out.vertex_normal.append(np.asarray((scan.normal_vertex[i])[:, vertex_idx]))
		#out.colors.append(np.asarray(scan.clouds[i].colors)[point_idx, :])

		# valid_points = []
		# if i == 0:
		# 	cnt = 0
		# 	for l in idx:
		# 		if l not in idx_valid:
		# 			out.labels[0][cnt] = 0
		# 		else:
		# 			valid_points.append(cnt)
		# 		cnt += 1

		point_idx_map = invert_index_map(point_idx)
		vertex_idx_map = invert_index_map(vertex_idx)
		point_idx_maps.append(point_idx_map)
		vertex_idx_maps.append(vertex_idx_map)
		out.point_index_maps.append(np.asarray(point_idx))
		out.vertex_index_maps.append(np.asarray(vertex_idx))

	# valid_labeled = np.count_nonzero(np.asarray(out.labels[0]))
	# print("Valid labeled: " + str(valid_labeled))

	for i in range(0, num_scales):
		ci_points = scan.conv_ind[i][:, out.point_index_maps[i]].T
		ci_vertex = scan.conv_ind_vertex[i][:, out.vertex_index_maps[i]].T
		ci_points = remap_indices(point_idx_maps[i], ci_points)
		ci_vertex = remap_indices(point_idx_maps[i], ci_vertex)

		out.point_conv_ind.append(ci_points)
		out.vertex_conv_ind.append(ci_vertex)

		if i > 0:
			pool_ind_points = scan.pool_ind[i][out.point_index_maps[i], :]
			pool_ind_vertex = scan.pool_ind_vertex[i][out.vertex_index_maps[i], :]
			pool_ind_points = remap_indices(point_idx_maps[i-1], pool_ind_points)
			pool_ind_vertex = remap_indices(vertex_idx_maps[i-1], pool_ind_vertex)
		else:
			pool_ind_points = None
			pool_ind_vertex = None
		out.vertex_pool_ind.append(pool_ind_vertex)
		out.point_pool_ind.append(pool_ind_points)

	return out

def get_test_vertex_scan_part_out(par, valid_rad, search_offset, batch_center=None, sample_type='POINT'):
	lb_count = 0
	num_scales = len(scan.clouds)

	point_idx_maps = []
	vertex_idx_maps = []
	out = BatchData()

	valid_min = batch_center - valid_rad
	valid_max = batch_center + valid_rad

	valid_vertex_idx = np.nonzero(np.asarray([(np.asarray(scan.vertex_clouds[0].points) <= valid_max).all(1),
				(np.asarray(scan.vertex_clouds[0].points) >= valid_min).all(1)]).all(0))[0]

	# [k, valid_vertex_idx, _] = scan.vertex_trees[0].search_radius_vector_3d(batch_center, radius=valid_rad)
	# valid_vertex_idx = np.asarray(valid_vertex_idx)
	num_vertex = valid_vertex_idx.shape[0]
	tmp_labels = np.asarray(scan.labels_gt[0][valid_vertex_idx])
	out.labels.append(tmp_labels)

	curr_w = np.zeros(num_vertex, np.float32)
	for lb_index in range(num_vertex):
		lb = tmp_labels[lb_index]
		if lb > 0:
			curr_w[lb_index] = par.d_par.class_weights[int(lb - 1)]
		else:
			curr_w[lb_index] = 0.0
	out.loss_weights.append(curr_w)

	for i in range(0, num_scales):
		search_min = batch_center - valid_rad - search_offset[i]
		search_max = batch_center + valid_rad + search_offset[i]
		if i == 0:
			vertex_idx = valid_vertex_idx
		else:
			vertex_idx = np.nonzero(
				np.asarray([(np.asarray(scan.vertex_clouds[i].points) <= search_max).all(1),
							(np.asarray(scan.vertex_clouds[i].points) >= search_min).all(1)]).all(0))[0]

		# vertex_idx = np.asarray(vertex_idx)
		point_idx = np.nonzero(
			np.asarray([(np.asarray(scan.clouds[i].points) <= search_max).all(1),
						(np.asarray(scan.clouds[i].points) >= search_min).all(1)]).all(0))[0]
		# [k, point_idx, _] = scan.trees[i].search_radius_vector_3d(batch_center, radius=valid_rad+search_offset[i])
		# point_idx = np.asarray(point_idx)
		tmp_vertex_num = vertex_idx.shape[0]
		tmp_point_num = point_idx.shape[0]

		out.point_depth.append(np.asarray((scan.depth[i])[:, point_idx]))
		out.vertex_depth.append(np.asarray((scan.depth_vertex[i])[:, vertex_idx]))
		out.point_normal.append(np.asarray((scan.normal[i])[:, point_idx]))
		out.vertex_normal.append(np.asarray((scan.normal_vertex[i])[:, vertex_idx]))
		# out.colors.append(np.asarray(scan.clouds[i].colors)[point_idx, :])

		# valid_points = []
		# if i == 0:
		# 	cnt = 0
		# 	for l in idx:
		# 		if l not in idx_valid:
		# 			out.labels[0][cnt] = 0
		# 		else:
		# 			valid_points.append(cnt)
		# 		cnt += 1

		point_idx_map = invert_index_map(point_idx)
		vertex_idx_map = invert_index_map(vertex_idx)
		point_idx_maps.append(point_idx_map)
		vertex_idx_maps.append(vertex_idx_map)
		out.point_index_maps.append(np.asarray(point_idx))
		out.vertex_index_maps.append(np.asarray(vertex_idx))

	# valid_labeled = np.count_nonzero(np.asarray(out.labels[0]))
	# print("Valid labeled: " + str(valid_labeled))

	for i in range(0, num_scales):
		ci_points = scan.conv_ind[i][:, out.point_index_maps[i]].T
		ci_vertex = scan.conv_ind_vertex[i][:, out.vertex_index_maps[i]].T
		ci_points = remap_indices(point_idx_maps[i], ci_points)
		ci_vertex = remap_indices(point_idx_maps[i], ci_vertex)

		out.point_conv_ind.append(ci_points)
		out.vertex_conv_ind.append(ci_vertex)

		if i > 0:
			pool_ind_points = scan.pool_ind[i][out.point_index_maps[i], :]
			pool_ind_vertex = scan.pool_ind_vertex[i][out.vertex_index_maps[i], :]
			pool_ind_points = remap_indices(point_idx_maps[i-1], pool_ind_points)
			pool_ind_vertex = remap_indices(vertex_idx_maps[i-1], pool_ind_vertex)
		else:
			pool_ind_points = None
			pool_ind_vertex = None
		out.vertex_pool_ind.append(pool_ind_vertex)
		out.point_pool_ind.append(pool_ind_points)

	return out

def get_vertex_batch_array(scan_var, par, points=None):
	global scan
	scan = scan_var
	num_cores = multiprocessing.cpu_count()

	if points is None:
		pts = []
		for i in range(0, par.batch_array_size):
			pt, ind = scan_var.get_random_partial_vertex(par.valid_rad)
			pts.append(pt)
		arr_size = par.batch_array_size
	else:
		pts = points
		arr_size = len(points)
		print(arr_size)

	if num_cores > 16:
		num_cores = 16


	batch_array = Parallel(n_jobs=num_cores)(
			delayed(get_vertex_scan_part_out)(par, pts[i]) for i in range(0, arr_size))
	#get_vertex_scan_part_out(par, pts[0])
	return batch_array

def get_test_vertex_batch_array(scan_var, par, search_offset, points=None, widthes=None):
	global scan
	scan = scan_var
	num_cores = multiprocessing.cpu_count()

	if points is None:
		pts = []
		for i in range(0, par.batch_array_size):
			pt, ind = scan_var.get_random_partial_vertex(par.valid_rad)
			pts.append(pt)
		arr_size = par.batch_array_size
	else:
		pts = points
		arr_size = len(points)
		print(arr_size)

	if num_cores > 16:
		num_cores = 16

	batch_array = Parallel(n_jobs=num_cores)(
			delayed(get_test_vertex_scan_part_out)(par, valid_rad=widthes[i],
												   search_offset=search_offset, batch_center=pts[i]) for i in range(0, arr_size))
	return batch_array


