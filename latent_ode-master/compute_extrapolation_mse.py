import lib.utils as utils
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

def compute_extrapolation_mse(model,
	test_dataloader,args,
	n_batches, experimentID, device,
	n_traj_samples = 1, kl_coef = 1.):

	mse=0
	total_points=0

	if (args.latent_ode or args.rnn_vae):

		for i in range(n_batches):
			print("Computing extrapolation mse... " + str(i))

			batch_dict = utils.get_next_batch(test_dataloader)
			pred_x, _ = model.get_reconstruction(batch_dict["tp_to_predict"],
				batch_dict["observed_data"], batch_dict["observed_tp"],
				mask = batch_dict["observed_mask"], n_traj_samples = n_traj_samples,
				mode = batch_dict["mode"])


			real_x=batch_dict["data_to_predict"][:,:,:-3]
			pred_x=torch.mean(pred_x,0)[:,:,:-3]
			mask_predicted_data=batch_dict["mask_predicted_data"][:,:,:-3]



			pred_x_masked=torch.masked_select(pred_x,mask_predicted_data.to(torch.bool))

			real_x_masked=torch.masked_select(real_x,mask_predicted_data.to(torch.bool))

			if(torch.numel(pred_x_masked)>0):
				mse=mse+mean_squared_error(real_x_masked.cpu().numpy(),pred_x_masked.cpu().numpy())*torch.numel(pred_x_masked)
				total_points=total_points+torch.numel(pred_x_masked)

	else:

		for i in range(n_batches):
			print("Computing extrapolation mse... " + str(i))

			batch_dict = utils.get_next_batch(test_dataloader)

			_, extra_info = model.get_reconstruction(batch_dict["observed_tp"],
				batch_dict["observed_data"], batch_dict["observed_tp"],
				mask = batch_dict["observed_mask"], n_traj_samples = n_traj_samples,
				mode = batch_dict["mode"])

			pred_x, _ = model.forwardextrap(batch_dict["tp_to_predict"], extra_info["last_hidden"], model.decoder(extra_info["last_hidden"]),
				n_traj_samples = n_traj_samples,
				mode = batch_dict["mode"])


			real_x=batch_dict["data_to_predict"][:,:,:-3]
			pred_x=torch.mean(pred_x,0)[:,:,:-3]
			mask_predicted_data=batch_dict["mask_predicted_data"][:,:,:-3]

			pred_x_masked=torch.masked_select(pred_x,mask_predicted_data.to(torch.bool))

			real_x_masked=torch.masked_select(real_x,mask_predicted_data.to(torch.bool))

			if(torch.numel(pred_x_masked)>0):
				mse=mse+mean_squared_error(real_x_masked.cpu().numpy(),pred_x_masked.cpu().numpy())*torch.numel(pred_x_masked)
				total_points=total_points+torch.numel(pred_x_masked)

	mse=mse/total_points
	return mse
