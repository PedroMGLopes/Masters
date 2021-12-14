import lib.utils as utils
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

def compute_interpolation_mse(model,
	test_dataloader,percentage, args,
	n_batches, experimentID, device,
	n_traj_samples = 1, kl_coef = 1.):

	mse=0
	total_random_points=0
	for i in range(n_batches):
		print("Computing interpolation mse... " + str(i))

		batch_dict = utils.get_next_batch(test_dataloader)

		random_points=torch.rand(batch_dict["observed_data"].size()).to(device)*batch_dict["observed_mask"]
		random_points=(random_points<percentage).int()

		pred_x, _ = model.get_reconstruction(batch_dict["tp_to_predict"],
			batch_dict["observed_data"]*(1-random_points), batch_dict["observed_tp"],
			mask = batch_dict["observed_mask"]-(random_points*batch_dict["observed_mask"]), n_traj_samples = n_traj_samples,
			mode = batch_dict["mode"])

		real_x=batch_dict["data_to_predict"][:,:,:-3]
		pred_x=torch.mean(pred_x,0)[:,:,:-3]
		random_points=random_points[:,:,:-3]
		observed_mask=batch_dict["observed_mask"][:,:,:-3]

		pred_x_masked=torch.masked_select(pred_x,(random_points*observed_mask).to(torch.bool))
		real_x_masked=torch.masked_select(real_x,(random_points*observed_mask).to(torch.bool))

		mse=mse+mean_squared_error(real_x_masked.cpu().numpy(),pred_x_masked.cpu().numpy())*torch.numel(pred_x_masked)
		total_random_points=total_random_points+torch.numel(pred_x_masked)
		
	mse=mse/total_random_points
	return mse
