import numpy as np
import tensorflow as tf
from time import gmtime, strftime, time
import random

from common import *
from cloud import *
from dataset_params import *
from point_ops import *
from general_ops import *
import simple_octree
# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
# os.system('rm tmp')
os.environ['CUDA_VISIBLE_DEVICES']="0"

class param:
	def __init__(self, config):
		# self.pre_output_dir = os.path.join(get_tc_path(), config['pre_output_dir'])
		# self.experiment_dir = os.path.join(get_tc_path(), config['co_experiment_dir'])
		self.pre_output_dir = config['pre_output_dir']
		self.experiment_dir = config['co_experiment_dir']
		self.output_dir = os.path.join(self.experiment_dir, config['co_output_dir'])
		# self.train_file = os.path.join(get_tc_path(), config['co_train_file'])
		# self.test_file = os.path.join(get_tc_path(), config['co_test_file'])
		self.train_file = config['co_train_file']
		self.test_file = config['co_test_file']
		self.num_rotations = config['pre_num_rotations']
		self.log_dir = os.path.join(self.experiment_dir, config['tt_log_dir'])
		self.snapshot_dir = os.path.join(self.experiment_dir, config['tt_snapshot_dir'])
		self.dataset_type = config['pre_dataset_param']
		if self.dataset_type == "stanford":
			self.d_par = stanford_params()
		elif self.dataset_type == "scannet":
			self.d_par = scannet_params()
		elif self.dataset_type == "semantic3d":
			self.d_par = semantic3d_params()
		elif self.dataset_type == "dtu_mvs":
			self.d_par = dtu_mvs_params()

		self.vertex_depth = config['pre_octree_depth']
		self.input_type = config['tt_input_type']
		self.max_snapshots = config['tt_max_snapshots']
		self.test_iter = config['tt_test_iter']
		self.reload_iter = config['tt_reload_iter']
		self.max_iter_count = config['tt_max_iter_count']
		self.batch_size = config['tt_batch_size']
		self.valid_rad_factor = config['tt_valid_rad_factor']
		# self.valid_rad_vertex = config['tt_valid_rad_vertex']
		self.filter_size = config['tt_filter_size']
		self.batch_array_size = config['tt_batch_array_size']

		# self.min_cube_size = config['pre_min_cube_size']
		self.min_cube_size_vertex = 1.0 / (2 ** config['pre_octree_depth'])
		self.min_cube_size_points = 1.0 / (2 ** (config['pre_octree_depth']+2))
		self.cube_size = [self.min_cube_size_vertex, 2*self.min_cube_size_vertex, 4*self.min_cube_size_vertex]
		self.conv_rad = 2 * np.asarray(self.cube_size)
		self.valid_rad = self.valid_rad_factor*self.min_cube_size_vertex
		self.num_scales = len(self.cube_size)
		self.data_sampling_type = config['tt_full_or_part']

	def full_rf_size(self):
		return self.valid_rad + 4*self.conv_rad[0] + 4*self.conv_rad[1] + 2*self.conv_rad[2]


class model():

	def __init__(self, curr_param):
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
		self.training_step = 0
		self.par = curr_param

	def load_data(self, mode):
		if mode == "train":
			file_name = self.par.train_file
			self.training_data = []
		else:
			file_name = self.par.test_file
			self.test_data = []

		with open(file_name) as f:
			scans = f.readlines()

		scans = [s.rstrip() for s in scans]
		scans = [s for s in scans if s]

		if mode == "train":
			self.training_scans = scans
		else:
			self.test_scans = scans

		scans = [os.path.join(self.par.pre_output_dir, s.rstrip()) for s in scans]

		cnt = 0
		for s_path in scans:
			s = ScanData()
			s.load(s_path, self.par)
			s.remap_depth(vmin=-self.par.conv_rad[0], vmax=self.par.conv_rad[0])
			if mode == "train":
				self.training_data.append(s)
			else:
				batchs_file = (os.path.join(s_path, "batch_centers.txt"))
				s.batches.append(np.loadtxt(batchs_file, skiprows=1))
				self.test_data.append(s)
			cnt += 1

	def precompute_vertex_validation_batches(self):
		self.vertex_validation_batches = []
		for test_scan in self.test_data:
			if self.par.data_sampling_type == 'part':
				vertex_batch_array = get_vertex_batch_array(test_scan, self.par)
				for b in vertex_batch_array:
					if np.shape(b.point_depth[0])[1] <= self.par.batch_size \
							and np.shape(b.vertex_depth[0])[1] <= self.par.batch_size:
						self.vertex_validation_batches.append(b)
			else:
				b = get_vertex_batch_from_full_scan(test_scan, self.par.num_scales, self.par.d_par.class_weights)
				if np.shape(b.point_depth[0])[1] <= self.par.batch_size \
						and np.shape(b.vertex_depth[0])[1] <= self.par.batch_size:
					self.vertex_validation_batches.append(b)
		print('valid_validation_batches_num', len(self.vertex_validation_batches))

	def get_vertex_training_batch(self, iter_num):
		if self.par.data_sampling_type == 'full':
			num_train_scans = len(self.training_data)
			scan_num = iter_num % num_train_scans
			return get_vertex_batch_from_full_scan(self.training_data[scan_num], self.par.num_scales,
											self.par.d_par.class_weights)
		else:
			scan_num = iter_num % (5*self.par.batch_array_size)
			if scan_num == 0:
				random_scan = random.randint(0, len(self.training_data) - 1)
				self.tr_batch_array = get_vertex_batch_array(self.training_data[random_scan], self.par)
			return self.tr_batch_array[scan_num % self.par.batch_array_size]

	def get_vertex_feed_dict(self, b):
		bs = self.par.batch_size

		mask1 = get_pooling_mask(b.point_pool_ind[1])
		mask2 = get_pooling_mask(b.point_pool_ind[2])

		ret_dict = {self.c1_ind: expand_dim_to_batch2(b.point_conv_ind[0], bs),
					self.c1_ind_vertex: expand_dim_to_batch2(b.vertex_conv_ind[0], bs),
					self.c2_ind: expand_dim_to_batch2(b.point_conv_ind[1], bs//2),
					self.c2_ind_vertex: expand_dim_to_batch2(b.vertex_conv_ind[1], bs//2),
					self.c3_ind: expand_dim_to_batch2(b.point_conv_ind[2], bs//4),
					self.c3_ind_vertex: expand_dim_to_batch2(b.vertex_conv_ind[2], bs//4),
					self.p12_ind: expand_dim_to_batch2(b.point_pool_ind[1], bs//2),
					self.p12_ind_vertex: expand_dim_to_batch2(b.vertex_pool_ind[1], bs//2),
					self.p12_mask: expand_dim_to_batch2(mask1, bs//2, dummy_val=0),
					self.p23_ind: expand_dim_to_batch2(b.point_pool_ind[2], bs//4),
					self.p23_ind_vertex: expand_dim_to_batch2(b.vertex_pool_ind[2], bs//4),
					self.p23_mask: expand_dim_to_batch2(mask2, bs//4, dummy_val=0),
					self.label: expand_dim_to_batch1(b.labels[0], bs),
					self.loss_weight: expand_dim_to_batch1(b.loss_weights[0], bs)}

		if 'd' in self.par.input_type:
			ret_dict.update({self.input_depth1: expand_dim_to_batch2(b.point_depth[0].T, bs)})
			ret_dict.update({self.input_depth2: expand_dim_to_batch2(b.point_depth[1].T, bs//2)})
			ret_dict.update({self.input_depth3: expand_dim_to_batch2(b.point_depth[2].T, bs//4)})
			ret_dict.update({self.input_vertex_depth1: expand_dim_to_batch2(b.vertex_depth[0].T, bs)})
			ret_dict.update({self.input_vertex_depth2: expand_dim_to_batch2(b.vertex_depth[1].T, bs // 2)})
			ret_dict.update({self.input_vertex_depth3: expand_dim_to_batch2(b.vertex_depth[2].T, bs // 4)})
		if 'n' in self.par.input_type:
			patch_size_pow2 = np.shape(b.point_depth[0])[0]
			# print('get_vertex_feed_dict patch_size_pow2', patch_size_pow2)
			point_normal0 = np.dstack((
				expand_dim_to_batch2(((b.point_normal[0])[0:patch_size_pow2, :]).T, bs),
				expand_dim_to_batch2(((b.point_normal[0])[patch_size_pow2:2*patch_size_pow2, :]).T, bs),
				expand_dim_to_batch2(((b.point_normal[0])[2*patch_size_pow2:3*patch_size_pow2, :]).T, bs)))
			point_normal1 = np.dstack((
				expand_dim_to_batch2(((b.point_normal[1])[0:patch_size_pow2, :]).T, bs//2),
				expand_dim_to_batch2(((b.point_normal[1])[patch_size_pow2:2*patch_size_pow2, :]).T, bs//2),
				expand_dim_to_batch2(((b.point_normal[1])[2*patch_size_pow2:3*patch_size_pow2, :]).T, bs//2)))
			point_normal2 = np.dstack((
				expand_dim_to_batch2(((b.point_normal[2])[0:patch_size_pow2, :]).T, bs//4),
				expand_dim_to_batch2(((b.point_normal[2])[patch_size_pow2:2*patch_size_pow2, :]).T, bs//4),
				expand_dim_to_batch2(((b.point_normal[2])[2*patch_size_pow2:3*patch_size_pow2, :]).T, bs//4)))
			ret_dict.update({self.input_normal1: point_normal0})
			ret_dict.update({self.input_normal2: point_normal1})
			ret_dict.update({self.input_normal3: point_normal2})

			vertex_normal0 = np.dstack((
				expand_dim_to_batch2(((b.vertex_normal[0])[0:patch_size_pow2, :]).T, bs),
				expand_dim_to_batch2(((b.vertex_normal[0])[patch_size_pow2:2*patch_size_pow2, :]).T, bs),
				expand_dim_to_batch2(((b.vertex_normal[0])[2*patch_size_pow2:3*patch_size_pow2, :]).T, bs)))
			vertex_normal1 = np.dstack((
				expand_dim_to_batch2(((b.vertex_normal[1])[0:patch_size_pow2, :]).T, bs//2),
				expand_dim_to_batch2(((b.vertex_normal[1])[patch_size_pow2:2*patch_size_pow2, :]).T, bs//2),
				expand_dim_to_batch2(((b.vertex_normal[1])[2*patch_size_pow2:3*patch_size_pow2, :]).T, bs//2)))
			vertex_normal2 = np.dstack((
				expand_dim_to_batch2(((b.vertex_normal[2])[0:patch_size_pow2, :]).T, bs//4),
				expand_dim_to_batch2(((b.vertex_normal[2])[patch_size_pow2:2*patch_size_pow2, :]).T, bs//4),
				expand_dim_to_batch2(((b.vertex_normal[2])[2*patch_size_pow2:3*patch_size_pow2, :]).T, bs//4)))
			ret_dict.update({self.input_vertex_normal1: vertex_normal0})
			ret_dict.update({self.input_vertex_normal2: vertex_normal1})
			ret_dict.update({self.input_vertex_normal3: vertex_normal2})

		if 'c' in self.par.input_type:
			ret_dict.update({self.input_colors: expand_dim_to_batch2(b.colors[0], bs)})

		return ret_dict

	def build_vertex_model(self, batch_size):
		self.best_accuracy = 0.0
		fs = self.par.filter_size
		bs = batch_size

		num_input_ch = 0
		input_list = []
		num_input_dn_ch = 0
		if 'd' in self.par.input_type:
			num_input_dn_ch += 1
			self.input_depth1 = tf.placeholder(tf.float32, [bs, fs * fs])
			self.input_depth2 = tf.placeholder(tf.float32, [bs // 2, fs * fs])
			self.input_depth3 = tf.placeholder(tf.float32, [bs // 4, fs * fs])
			self.input_vertex_depth1 = tf.placeholder(tf.float32, [bs, fs * fs])
			self.input_vertex_depth2 = tf.placeholder(tf.float32, [bs // 2, fs * fs])
			self.input_vertex_depth3 = tf.placeholder(tf.float32, [bs // 4, fs * fs])
		if 'n' in self.par.input_type:
			num_input_dn_ch += 3
			self.input_normal1 = tf.placeholder(tf.float32, [bs, fs * fs, 3])
			self.input_normal2 = tf.placeholder(tf.float32, [bs // 2, fs * fs, 3])
			self.input_normal3 = tf.placeholder(tf.float32, [bs // 4, fs * fs, 3])
			self.input_vertex_normal1 = tf.placeholder(tf.float32, [bs, fs * fs, 3])
			self.input_vertex_normal2 = tf.placeholder(tf.float32, [bs // 2, fs * fs, 3])
			self.input_vertex_normal3 = tf.placeholder(tf.float32, [bs // 4, fs * fs, 3])
		if 'c' in self.par.input_type:
			num_input_ch += 3
			self.input_colors = tf.placeholder(tf.float32, [bs, 3])
			input_list.append(self.input_colors)

		self.c1_ind = tf.placeholder(tf.int32, [bs, fs * fs])
		self.c1_ind_vertex = tf.placeholder(tf.int32, [bs, fs * fs])
		self.p12_ind = tf.placeholder(tf.int32, [bs // 2, 8])
		self.p12_ind_vertex = tf.placeholder(tf.int32, [bs // 2, 8])
		self.p12_mask = tf.placeholder(tf.float32, [bs // 2, 8])
		# self.p12_mask_vertex = tf.placeholder(tf.float32, [bs // 2, 8])
		self.c2_ind = tf.placeholder(tf.int32, [bs // 2, fs * fs])
		self.c2_ind_vertex = tf.placeholder(tf.int32, [bs // 2, fs * fs])
		self.p23_ind = tf.placeholder(tf.int32, [bs // 4, 8])
		self.p23_ind_vertex = tf.placeholder(tf.int32, [bs // 4, 8])
		self.p23_mask = tf.placeholder(tf.float32, [bs // 4, 8])
		# self.p23_mask_vertex = tf.placeholder(tf.float32, [bs // 4, 8])
		self.c3_ind = tf.placeholder(tf.int32, [bs // 4, fs * fs])
		self.c3_ind_vertex = tf.placeholder(tf.int32, [bs // 4, fs * fs])

		# self.scatter_ind = tf.placeholder(tf.int32, [bs, 12])

		# self.gather_ind = tf.placeholder(tf.int32, [bs, 2])

		self.label = tf.placeholder(tf.int32, [bs])
		self.loss_weight = tf.placeholder(tf.float32, [bs])

		label_mask = tf.cast(self.label, tf.bool)

		shape_unpool2 = tf.constant([bs // 2, 64])
		shape_unpool1 = tf.constant([bs, 32])

		shape_scatter = tf.constant([bs, 32])

		input_dn_list1 = []
		if 'd' in self.par.input_type:
			input_depth1 = tf.expand_dims(self.input_depth1, axis=2)
			input_dn_list1.append(input_depth1)

		if 'n' in self.par.input_type:
			input_dn_list1.append(self.input_normal1)

		if input_dn_list1:
			dn_input1 = tf.concat(input_dn_list1, axis=2)
			if num_input_ch > 0:
				signal_input = tf.concat(input_list, axis=1)
				h_conv1 = lrelu(point_conv2('conv1', signal_input, self.c1_ind,
											fs * fs, num_input_ch+num_input_dn_ch,
											32, extra_chans=dn_input1))
			else:
				signal_input = tf.expand_dims(dn_input1, axis=0)
				h_conv1 = lrelu(conv_2d_layer('conv1', signal_input, num_input_dn_ch, 32, 1,
											  fs * fs, 1, 1, padding='VALID'))
		else:
			signal_input = tf.concat(input_list, axis=1)
			h_conv1 = lrelu(point_conv('conv1', signal_input, self.c1_ind,
									   fs * fs, num_input_ch, 32))

		h_conv1 = tf.squeeze(h_conv1)
		h_conv11 = lrelu(point_conv('conv11', h_conv1, self.c1_ind,
									fs * fs, 32, 32))

		h_pool1 = point_pool(h_conv11, self.p12_ind, self.p12_mask)

		input_dn_list2 = []
		if 'd' in self.par.input_type:
			input_depth2 = tf.expand_dims(self.input_depth2, axis=2)
			input_dn_list2.append(input_depth2)

		if 'n' in self.par.input_type:
			input_dn_list2.append(self.input_normal2)

		if input_dn_list2:
			dn_input2 = tf.concat(input_dn_list2, axis=2)
			h_conv2 = lrelu(point_conv2('conv2', h_pool1, self.c2_ind, fs * fs, 32+num_input_dn_ch, 64,
									   extra_chans=dn_input2))
		else:
			h_conv2 = lrelu(point_conv('conv2', h_pool1, self.c2_ind, fs * fs, 32, 64))
		h_conv22 = lrelu(point_conv('conv22', h_conv2, self.c2_ind, fs * fs, 64, 64))

		h_pool2 = point_pool(h_conv22, self.p23_ind, self.p23_mask)

		input_dn_list3 = []
		if 'd' in self.par.input_type:
			input_depth3 = tf.expand_dims(self.input_depth3, axis=2)
			input_dn_list3.append(input_depth3)

		if 'n' in self.par.input_type:
			input_dn_list3.append(self.input_normal3)

		if input_dn_list3:
			dn_input3 = tf.concat(input_dn_list3, axis=2)
			h_conv3 = lrelu(point_conv2('conv3', h_pool2, self.c3_ind, fs * fs, 64+num_input_dn_ch, 128,
									   extra_chans=dn_input3))
		else:
			h_conv3 = lrelu(point_conv('conv3', h_pool2, self.c3_ind, fs * fs, 64, 128))
		h_conv33 = lrelu(point_conv('conv33', h_conv3, self.c3_ind, fs * fs, 128, 64))

		s_h_conv33 = tf.squeeze(h_conv33)


		input_vdn_list3 = []
		if 'd' in self.par.input_type:
			input_vertex_depth3 = tf.expand_dims(self.input_vertex_depth3, axis=2)
			input_vdn_list3.append(input_vertex_depth3)

		if 'n' in self.par.input_type:
			input_vdn_list3.append(self.input_vertex_normal3)

		if input_vdn_list3:
			vdn_input3 = tf.concat(input_vdn_list3, axis=2)
			vertex_conv3 = lrelu(point_conv2('v_conv3', s_h_conv33, self.c3_ind_vertex, fs * fs, 64+num_input_dn_ch, 128,
											 extra_chans=vdn_input3))
		else:
			vertex_conv3 = lrelu(point_conv('v_conv3', s_h_conv33, self.c3_ind_vertex, fs * fs, 64, 128))

		vertex_conv3 = tf.expand_dims(tf.expand_dims(vertex_conv3, axis=1), axis=0)
		vertex_conv33 = tf.squeeze(lrelu(conv_2d_layer('v_conv33', vertex_conv3, 128, 64, 1, 1, 1, 1)))

		vertex_unpool2 = point_unpool(vertex_conv33, self.p23_ind_vertex, shape_unpool2)

		h_unpool2 = point_unpool(h_conv33, self.p23_ind, shape_unpool2)
		uconv2_in = tf.concat([h_conv22, h_unpool2], axis=1)
		h_uconv2 = lrelu(point_conv('uconv2', uconv2_in, self.c2_ind, fs * fs, 128, 64))
		h_uconv22 = lrelu(point_conv('uconv22', h_uconv2, self.c2_ind, fs * fs, 64, 32))

		s_h_uconv22 = tf.squeeze(h_uconv22)

		input_vdn_list2 = []
		if 'd' in self.par.input_type:
			input_vertex_depth2 = tf.expand_dims(self.input_vertex_depth2, axis=2)
			input_vdn_list2.append(input_vertex_depth2)

		if 'n' in self.par.input_type:
			input_vdn_list2.append(self.input_vertex_normal2)

		if input_vdn_list2:
			vdn_input2 = tf.concat(input_vdn_list2, axis=2)
			vertex_conv2 = lrelu(point_conv2('v_conv2', s_h_uconv22, self.c2_ind_vertex, fs * fs, 32+num_input_dn_ch, 64,
									   extra_chans=vdn_input2))
		else:
			vertex_conv2 = lrelu(point_conv('v_conv2', s_h_uconv22, self.c2_ind_vertex, fs * fs, 32, 64))

		vertex_conv2 = tf.expand_dims(tf.expand_dims(vertex_conv2, axis=1), axis=0)
		vertex_conv22 = tf.squeeze(lrelu(conv_2d_layer('v_conv22', vertex_conv2, 64, 64, 1, 1, 1, 1)))

		vertex_uconv2_in = tf.concat([vertex_conv22, vertex_unpool2], axis=1)
		vertex_uconv2_in = tf.expand_dims(tf.expand_dims(vertex_uconv2_in, axis=1), axis=0)
		vertex_uconv2 = lrelu(conv_2d_layer('v_uconv2', vertex_uconv2_in, 128, 64, 1, 1, 1, 1))
		# vertex_uconv2 = tf.expand_dims(tf.expand_dims(vertex_uconv2, axis=1), axis=0)
		vertex_uconv22 = tf.squeeze(lrelu(conv_2d_layer('v_uconv22', vertex_uconv2, 64, 32, 1, 1, 1, 1)))
		vertex_unpool1 = point_unpool(vertex_uconv22, self.p12_ind_vertex, shape_unpool1)

		h_unpool1 = point_unpool(h_uconv22, self.p12_ind, shape_unpool1)
		uconv1_in = tf.concat([h_conv11, h_unpool1], axis=1)
		h_uconv1 = lrelu(point_conv('uconv1', uconv1_in, self.c1_ind, fs * fs, 64, 32))
		h_uconv11 = tf.squeeze(point_conv('uconv11', h_uconv1, self.c1_ind, fs * fs, 32, 32))

		s_h_conv11 = tf.squeeze(lrelu(h_uconv11))

		input_vdn_list1 = []
		if 'd' in self.par.input_type:
			input_vertex_depth1 = tf.expand_dims(self.input_vertex_depth1, axis=2)
			input_vdn_list1.append(input_vertex_depth1)

		if 'n' in self.par.input_type:
			input_vdn_list1.append(self.input_vertex_normal1)

		if input_vdn_list1:
			vdn_input1 = tf.concat(input_vdn_list1, axis=2)
			vertex_conv1 = lrelu(point_conv2('v_conv1', s_h_conv11, self.c1_ind_vertex, fs * fs, 32+num_input_dn_ch, 64,
									   extra_chans=vdn_input1))
		else:
			vertex_conv1 = lrelu(point_conv('v_conv1', s_h_conv11, self.c1_ind_vertex, fs * fs, 32, 64))

		vertex_conv1 = tf.expand_dims(tf.expand_dims(vertex_conv1, axis=1), axis=0)
		vertex_conv11 = tf.squeeze(lrelu(conv_2d_layer('v_conv11', vertex_conv1, 64, 32, 1, 1, 1, 1)))

		vertex_uconv1_in = tf.concat([vertex_conv11, vertex_unpool1], axis=1)
		vertex_uconv1_in = tf.expand_dims(tf.expand_dims(vertex_uconv1_in, axis=1), axis=0)
		vertex_uconv1 = lrelu(conv_2d_layer('v_uconv1', vertex_uconv1_in, 64, 32, 1, 1, 1, 1))
		# vertex_uconv1 = tf.expand_dims(tf.expand_dims(vertex_uconv1, axis=1), axis=0)
		vertex_uconv11 = tf.squeeze(lrelu(conv_2d_layer('v_uconv11', vertex_uconv1, 32, 32, 1, 1, 1, 1)))

		pred_input = tf.expand_dims(tf.expand_dims(vertex_uconv11, axis=1), axis=0)
		v_pred = tf.squeeze(conv_2d_layer('pred1', pred_input, 32, self.par.d_par.num_classes - 1, 1, 1, 1, 1))

		self.output_label = tf.argmax(v_pred, axis=1, output_type=tf.int32)
		self.output_softmax = tf.nn.softmax(v_pred, axis=1)
		self.output_pred = v_pred

		masked_output = tf.boolean_mask(v_pred, label_mask)
		masked_label = tf.boolean_mask(self.label, label_mask)
		masked_weights = tf.boolean_mask(self.loss_weight, label_mask)

		real_label = tf.subtract(masked_label, 1)

		label2_mask = tf.cast(real_label, tf.bool)
		label1_mask = tf.equal(real_label, 0)

		label2_output = tf.boolean_mask(masked_output, label2_mask)
		label2_label = tf.boolean_mask(real_label, label2_mask)

		label1_output = tf.boolean_mask(masked_output, label1_mask)
		label1_label = tf.boolean_mask(real_label, label1_mask)

		tr_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "")

		self.loss = tf.reduce_mean(tf.multiply(masked_weights,
											   tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real_label,
																							  logits=masked_output)))
		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(0.001, global_step,
												   100000, 0.2, staircase=True)
		self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

		correct_prediction = tf.equal(tf.argmax(masked_output, axis=1, output_type=tf.int32), real_label)
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		label2_correct_prediction = tf.equal(tf.argmax(label2_output, axis=1, output_type=tf.int32), label2_label)
		self.label2_accuracy = tf.reduce_mean(tf.cast(label2_correct_prediction, tf.float32))

		label1_correct_prediction = tf.equal(tf.argmax(label1_output, axis=1, output_type=tf.int32), label1_label)
		self.label1_accuracy = tf.reduce_mean(tf.cast(label1_correct_prediction, tf.float32))

		self.test_loss_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
		self.test_loss_summary = tf.summary.scalar("accuracy", self.test_loss_placeholder)

		self.train_loss_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
		self.train_loss_summary = tf.summary.scalar("train_loss", self.train_loss_placeholder)

		curr_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
		self.writer = tf.summary.FileWriter(os.path.join(self.par.log_dir, curr_time))

		self.saver = tf.train.Saver(tr_var, max_to_keep=self.par.max_snapshots)

	def initialize_model(self):
		self.sess.run(tf.global_variables_initializer())

	def save_snapshot(self):
		self.saver.save(self.sess, os.path.join(self.par.snapshot_dir, 'model'),
			global_step=self.training_step)

	def load_snapshot(self):
		snapshot_name = tf.train.latest_checkpoint(self.par.snapshot_dir)
		if snapshot_name is not None:
			model_file_name = os.path.basename(snapshot_name)
			print("Loading snapshot " + model_file_name)
			itn = int(model_file_name.split('-')[1])
			self.training_step = itn
			self.saver.restore(self.sess, snapshot_name)

	def train(self):
		bs = self.par.batch_size
		for iter_i in range(self.training_step, self.par.max_iter_count):
			# if (iter_i > 0) and (iter_i % self.par.reload_iter == 0):
			# 	self.load_data("train")

			if iter_i % self.par.test_iter == 0:
				self.validate(iter_i)

			b = self.get_vertex_training_batch(iter_i)


			# if b.num_points() > bs:
			# 	continue
			if not b.check_valid(bs):
				continue

			out = self.sess.run([self.train_step, self.loss, self.output_label, self.accuracy, self.label2_accuracy,
					self.label1_accuracy], feed_dict=self.get_vertex_feed_dict(b))
			print(str(iter_i) + " : " + "loss: ", str(out[1]), "accuracy: ", str(out[3]), "label2_acc: ",
				  str(out[4]), "label1_acc: ", str(out[5]))

			summary = self.sess.run(self.train_loss_summary,
				feed_dict={self.train_loss_placeholder: out[1]})
			self.writer.add_summary(summary, iter_i)
			self.writer.add_graph(self.sess.graph, iter_i)

	def validate(self, step):
		pixel_count = 0
		acc = []
		pix = []
		label2_acc = []
		label1_acc = []
		bs = self.par.batch_size
		for b in self.vertex_validation_batches:
			out = self.sess.run([self.accuracy, self.output_label, self.label2_accuracy, self.label1_accuracy],
								feed_dict=self.get_vertex_feed_dict(b))
			valid_out = np.multiply(out[1], np.asarray(expand_dim_to_batch1(b.labels[0], bs), dtype=bool))
			acc.append(out[0])
			label2_acc.append(out[2])
			label1_acc.append(out[3])
			pix.append(np.count_nonzero(b.labels[0]))
			pixel_count += np.count_nonzero(b.labels[0])

		avg_acc = 0.0
		avg_label2_acc = 0.0
		avg_label1_acc = 0.0
		for i in range(0, len(acc)):
			avg_acc += acc[i] * pix[i] / pixel_count
			avg_label2_acc += label2_acc[i]
			avg_label1_acc += label1_acc[i]
		print("Accuracy: " + str(avg_acc), "label2_accuracy: ", str(avg_label2_acc/len(acc)),
			  "label1_accuracy: ", str(avg_label1_acc/len(acc)))

		if avg_acc > self.best_accuracy:
			self.best_accuracy = avg_acc
			self.save_snapshot()

		summary = self.sess.run(self.test_loss_summary,
			feed_dict={self.test_loss_placeholder: avg_acc})
		self.writer.add_summary(summary, step)

	def test(self):
		scan_id = 0
		print("Testing...")
		for val_scan in self.test_data:
			if self.par.data_sampling_type == 'full':
				scan_batches = [get_vertex_batch_from_full_scan(val_scan, self.par.num_scales, self.par.d_par.class_weights)]
				for b in scan_batches:
					out = self.sess.run([self.output_label, self.output_softmax, self.output_pred], feed_dict=self.get_vertex_feed_dict(b))
					# valid_out = np.multiply(out, np.asarray(expand_dim_to_batch1(b.labels[0], self.par.batch_size), dtype=bool))
					if self.par.data_sampling_type == 'full':
						val_scan.assign_vertex_labels_ratios_pres(out[0], out[1], out[2], np.shape(b.labels[0])[0])
			else:
				global scan
				min_bound = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
				max_bound = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
				scan = val_scan

				vertex_cube_size = np.array([self.par.min_cube_size_vertex, 2 * self.par.min_cube_size_vertex, 4 * self.par.min_cube_size_vertex])
				point_cube_size = np.array([self.par.min_cube_size_points, 2 * self.par.min_cube_size_points, 4 * self.par.min_cube_size_points])
				full_rf_offset = np.zeros(3)
				full_rf_offset[0] = 2 * self.par.conv_rad[0]
				full_rf_offset[1] = max(vertex_cube_size[1] + 2 * self.par.conv_rad[1],
										self.par.conv_rad[0] + point_cube_size[1] + self.par.conv_rad[1])
				full_rf_offset[2] = max(vertex_cube_size[1] + vertex_cube_size[2] + 2 * self.par.conv_rad[2],
										self.par.conv_rad[0] + point_cube_size[1] + point_cube_size[2] + self.par.conv_rad[2])

				full_rf_offset[0] = 2 * self.par.conv_rad[2]
				full_rf_offset[1] = 2 * self.par.conv_rad[2]
				full_rf_offset[2] = 2 * self.par.conv_rad[2]

				print('full_rf_offset ', full_rf_offset)

				obj_data_sets = []
				obj_data_sets.append(np.asarray(scan.vertex_clouds[0].points))
				obj_data_sets.append(np.asarray(scan.clouds[0].points))
				obj_data_sets.append(np.asarray(scan.vertex_clouds[1].points))
				obj_data_sets.append(np.asarray(scan.clouds[1].points))
				obj_data_sets.append(np.asarray(scan.vertex_clouds[2].points))
				obj_data_sets.append(np.asarray(scan.clouds[2].points))

				for data in obj_data_sets:
					print(data.shape)

				test_octree = simple_octree.Octree(1.0, np.array([0.0, 0.0, 0.0], dtype=np.float64))
				test_octree.divide_octree(obj_data_sets, full_rf_offset, self.par.batch_size)
				leaf_nodes = test_octree.gather_leaf_node()
				no_zero_leaf_node = [node for node in leaf_nodes if node.obj_nums[0] > 0]

				batch_centers = [node.position for node in no_zero_leaf_node]
				batch_widthes = [node.half_dimension for node in no_zero_leaf_node]

				print("Number of test batches: " + str(len(no_zero_leaf_node)))
				print("loading batches ...")
				scan_batches = get_test_vertex_batch_array(val_scan, self.par, full_rf_offset, batch_centers, batch_widthes)
				print('Done')
				for b_index, b in enumerate(scan_batches):
					print("batch_num", b_index)
					out = self.sess.run([self.output_label, self.output_softmax, self.output_pred],
										feed_dict=self.get_vertex_feed_dict(b))
					val_scan.assign_vertex_labels_ratios_pres_part(out[0], out[1], out[2], b.vertex_index_maps[0])

			make_dir(os.path.join(self.par.output_dir, self.test_scans[scan_id]))
			np.savetxt(os.path.join(self.par.output_dir, self.test_scans[scan_id], "pre_label.txt"),

					   val_scan.labels_pr[0], fmt='%d')
			np.savetxt(os.path.join(self.par.output_dir, self.test_scans[scan_id], "pre_softmax.txt"),
					   val_scan.ratios_pr[0], fmt='%f')
			np.savetxt(os.path.join(self.par.output_dir, self.test_scans[scan_id], "pre_before_softmax.txt"),
					   val_scan.pre_pr[0], fmt='%f')
			print(self.test_scans[scan_id])
			scan_id += 1

def run_net(config, mode):
	par = param(config)
	tf.reset_default_graph()
	nn = model(par)

	make_dir(par.log_dir)
	make_dir(par.output_dir)
	make_dir(par.snapshot_dir)


	if mode == "train":
		nn.load_data("test")
		nn.load_data("train")
		nn.build_vertex_model(par.batch_size)
		nn.precompute_vertex_validation_batches()
		nn.initialize_model()
		nn.load_snapshot()
		nn.train()
	elif mode == "test":
		nn.load_data("test")
		nn.build_vertex_model(par.batch_size)
		nn.initialize_model()
		nn.load_snapshot()
		nn.test()

	return 0
