import os
import matplotlib
if os.path.exists("/Users/yulia"):
	matplotlib.use('TkAgg')
else:
	matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
import pandas as pd
import lib.utils as utils
import numpy as np
import zipfile
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from lib.utils import get_device
import random

# Adapted from: https://github.com/rtqichen/time-series-datasets

 #if all_subsequences=True then cloning is ordered  else its shuffled
def clone_subsequences(data,all_subsequences=True):
	n_sequences=len(data)
	new_data=[]
	if(all_subsequences):
		position=0
		positionh=0
		for i in range(0,n_sequences):
			sequence_save=[]
			new_sequence=[0,0,0,0,0]
			new_sequence[0]=data[i][0]
			new_sequence[1]=data[i][1]
			new_sequence[2]=data[i][2]
			new_sequence[3]=data[i][3]
			new_sequence[4]=data[i][4]
			new_sequence_tuple=tuple(new_sequence)
			sequence_save.insert(0,new_sequence_tuple)
			for j in data[i][1][1:]:
				#remove last time step
				new_sequence[1]=new_sequence[1][:-1]
				#remove data of last time step
				new_sequence[2]=new_sequence[2][:-1,0:]
				new_sequence[3]=new_sequence[3][:-1,0:]
				#remove last label
				new_sequence[4]=new_sequence[4][:-1]
				new_sequence_tuple=tuple(new_sequence)
				sequence_save.insert(0,new_sequence_tuple)
			new_data=new_data+sequence_save
	else:
		for i in range(0,n_sequences):
			new_sequence=[0,0,0,0,0]
			new_sequence[0]=data[i][0]
			new_sequence[1]=data[i][1]
			new_sequence[2]=data[i][2]
			new_sequence[3]=data[i][3]
			new_sequence[4]=data[i][4]
			if(new_sequence[4].max()>0):
				while len(new_sequence[4])>1 and new_sequence[4][-1]>=0:
					#remove last time step
					new_sequence[1]=new_sequence[1][:-1]
					#remove data of last time step
					new_sequence[2]=new_sequence[2][:-1,0:]
					new_sequence[3]=new_sequence[3][:-1,0:]
					#remove last label
					new_sequence[4]=new_sequence[4][:-1]
					new_sequence_tuple=tuple(new_sequence)
					new_data.append(new_sequence_tuple)
		random.shuffle(new_data)
	print('done cloning')
	return new_data


# get minimum and maximum for each feature across the whole dataset
def get_data_min_max(records):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	data_min, data_max = None, None
	inf = torch.Tensor([float("Inf")])[0].to(device)

	for b, (record_id, tt, vals, mask, labels) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

	return data_min, data_max


class PhysioSepsis(object):

	urls = [
		'https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip',
		'https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip',
	]

	params = [
		'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2',
		'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
		'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
		'Age', 'Gender', 'HospAdmTime', 'utility'
	]

	params_dict = {k: i for i, k in enumerate(params)}

	def __init__(self, root, train=True, download=False,
		quantization = 0.1, n_samples = None, device = torch.device("cpu")):

		self.root = root
		self.train = train
		self.reduce = "average"
		self.quantization = quantization

		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found. You can use download=True to download it')

		if self.train:
			data_file = self.training_file
		else:
			data_file = self.test_file

		print(os.path.join(self.processed_folder, data_file))
		if device == torch.device("cpu"):
			print('cpu')
			self.data = torch.load(os.path.join(self.processed_folder, data_file), map_location='cpu')
		else:
			print('not cpu')
			self.data = torch.load(os.path.join(self.processed_folder, data_file))

		if n_samples is not None:
			self.data = self.data[:n_samples]



	def download(self):
		if self._check_exists():
			return

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		os.makedirs(self.raw_folder, exist_ok=True)
		os.makedirs(self.processed_folder, exist_ok=True)
		record_id = 0


		for url in self.urls:


			print('enter download')
			filename = url.rpartition('/')[2]
			if not os.path.exists(self.raw_folder + '/' +filename.split('.')[0]):
				download_url(url, self.raw_folder, filename, None)
				zipf = zipfile.ZipFile(os.path.join(self.raw_folder, filename))
				print(os.path.join(self.raw_folder, filename))
				zipf.extractall(self.raw_folder)
				print(self.raw_folder)
				zipf.close()

				if os.path.isdir(self.raw_folder + '/training' ):
					os.rename(self.raw_folder + '/training',self.raw_folder + '/' +filename.split('.')[0])

			print('Processing {}...'.format(filename))
			dirname = os.path.join(self.raw_folder, filename.split('.')[0])
			patients = []

			for file in os.listdir(dirname):

				id_df = pd.read_csv(dirname + '/' + file, sep='|')

				if id_df['ICULOS'][0]==1:
					labels=id_df['SepsisLabel']
					tt = torch.tensor(id_df['ICULOS']).to(self.device)
					vals = torch.from_numpy(id_df.drop(['SepsisLabel','ICULOS','Unit1','Unit2'],axis=1).applymap(lambda x: 0 if np.isnan(x) else x).to_numpy()).to(self.device)
					mask = torch.from_numpy(id_df.drop(['SepsisLabel','ICULOS','Unit1','Unit2'],axis=1).applymap(lambda x: 0 if np.isnan(x) else 1).to_numpy()).to(self.device)

					zeros = np.zeros(shape=(len(id_df['SepsisLabel'])))
					ones = np.ones(shape=(len(id_df['SepsisLabel'])))
					#Get scores for predicting zero or 1
					zeros_pred = compute_prediction_utility(labels.values, zeros, return_all_scores=True)
					ones_pred = compute_prediction_utility(labels.values, ones, return_all_scores=True)
					utility=torch.tensor((ones_pred-zeros_pred))

					patients.append((record_id, tt, vals, mask, utility))

					record_id=record_id+1

			torch.save(
				patients,
				os.path.join(self.processed_folder,
					filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
			)

		print('Done!')

	def _check_exists(self):
		for url in self.urls:
			filename = url.rpartition('/')[2]
			if not os.path.exists(
				os.path.join(self.processed_folder,
					filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
			):
				return False
		return True

	@property
	def raw_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'processed')

	@property
	def training_file(self):
		return 'training_setA_{}.pt'.format(self.quantization)

	@property
	def test_file(self):
		return 'training_setB_{}.pt'.format(self.quantization)

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
		fmt_str += '    Root Location: {}\n'.format(self.root)
		fmt_str += '    Quantization: {}\n'.format(self.quantization)
		fmt_str += '    Reduce: {}\n'.format(self.reduce)
		return fmt_str

	def visualize(self, timesteps, data, mask, plot_name):
		width = 15
		height = 15

		non_zero_attributes = (torch.sum(mask,0) > 2).numpy()
		non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
		n_non_zero = sum(non_zero_attributes)

		mask = mask[:, non_zero_idx]
		data = data[:, non_zero_idx]

		params_non_zero = [self.params[i] for i in non_zero_idx]
		params_dict = {k: i for i, k in enumerate(params_non_zero)}

		n_col = 3
		n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
		fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height), facecolor='white')

		#for i in range(len(self.params)):
		for i in range(n_non_zero):
			param = params_non_zero[i]
			param_id = params_dict[param]

			tp_mask = mask[:,param_id].long()

			tp_cur_param = timesteps[tp_mask == 1.]
			data_cur_param = data[tp_mask == 1., param_id]

			ax_list[i // n_col, i % n_col].plot(tp_cur_param.numpy(), data_cur_param.numpy(),  marker='o')
			ax_list[i // n_col, i % n_col].set_title(param)

		fig.tight_layout()
		fig.savefig(plot_name)
		plt.close(fig)


def variable_time_collate_fn_sepsis(batch, args, device = torch.device("cpu"), data_type = "train",
	data_min = None, data_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	D = batch[0][2].shape[1]
	N = 1 # number of labels
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	combined_tt = combined_tt.to(device)
	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_labels = torch.zeros([len(batch), len(combined_tt), N]).to(device)
	for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
		tt = tt.to(device)
		vals = vals.to(device)
		mask = mask.to(device)
		labels = labels.to(device)
		indices = inverse_indices[offset:offset + len(tt)]
		offset += len(tt)
		combined_vals[b, indices] = vals.float()
		combined_mask[b, indices] = mask.float()
		combined_labels[b, indices] = (labels.float()).unsqueeze(1)

	combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask,
		att_min = data_min, att_max = data_max)
	combined_vals=combined_vals.float()
	combined_tt = combined_tt.float()/336
	data_dict = {
		"data": combined_vals,
		"time_steps": combined_tt,
		"mask": combined_mask,
		"labels": combined_labels}

	data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
	return data_dict


def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0, check_errors=True, return_all_scores=False):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis?
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    if return_all_scores:
        return u
    else:
        return np.sum(u)


if __name__ == '__main__':
	torch.manual_seed(1991)

	dataset = PhysioSepsis('data/physiosepsis', train=False, download=True)
	dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_fn_sepsis)
	print(dataloader.__iter__().next())
