from physiosepsis import compute_prediction_utility
import lib.utils as utils
import numpy as np
import torch
from sklearn.metrics import accuracy_score

def compute_spesis_score(model,
	test_dataloader, args,
	n_batches, experimentID, device,
	n_traj_samples = 1, kl_coef = 1.,
	max_samples_for_eval = None,
	uncloned_test=None,
	cloned_test=None):

	dt_early=-12,
	dt_optimal=-6,
	dt_late=3.0,
	max_u_tp=1,
	min_u_fn=-2,
	u_fp=-0.05,
	u_tn=0,

	if (args.latent_ode or args.rnn_vae):

		classif_predictions = []
		all_test_labels =  []
		observed_utilities=[]
		best_utilities=[]
		inaction_utilities=[]
		acc_list=[]
		truep=0
		falsep=0
		falsen=0
		for i in range(n_batches):

			print("Computing sepsis score... " + str(i))

			batch_dict = utils.get_next_batch(test_dataloader)

			results  = model.compute_all_losses(batch_dict,
				n_traj_samples = n_traj_samples, kl_coef = kl_coef)


			n_labels = model.n_labels #batch_dict["labels"].size(-1)
			n_traj_samples = results["label_predictions"].size(0)
			for i in range(results["label_predictions"].size(1)):
				classif_predictions.append(results["label_predictions"][:,i,:,:].cpu()),
				all_test_labels.append(batch_dict["labels"][i,:,:].cpu())


		predicted=[]
		truevalue=[]
		counter_b=0
		for i in range(len(uncloned_test)):
			predicted_sample=[]
			true_value_sample=[]
			for k in range(n_traj_samples):
				predicted_sample.append([])
			for j in uncloned_test[i][1].cpu().detach().numpy():
				for k in range(n_traj_samples):
					classif_predictions[counter_b][k,j-1,0]=classif_predictions[counter_b][k,j-1,0]>=0
					classif_predictions[counter_b][k,j-1,0]=classif_predictions[counter_b][k,j-1,0].int()
					predicted_sample[k].append(classif_predictions[counter_b][k,j-1,0])
				all_test_labels[counter_b][j-1][0]=all_test_labels[counter_b][j-1][0]>=0
				all_test_labels[counter_b][j-1][0]=all_test_labels[counter_b][j-1][0].int()
				true_value_sample.append(all_test_labels[counter_b][j-1][0])
				counter_b+=1

			predicted.append(predicted_sample)
			truevalue.append(true_value_sample)


		for i in range(len(truevalue)):
			num_rows=len(truevalue[i])
			best_predictions     = np.zeros(num_rows)
			inaction_predictions = np.zeros(num_rows)
			labels=np.array(truevalue[i],dtype=np.float32)
			if np.any(labels):
				labels[labels.argmax()]=0
				labels[labels.argmax()]=0
				labels[labels.argmax()]=0
				labels[labels.argmax()]=0
				labels[labels.argmax()]=0
				labels[labels.argmax()]=0

			if np.any(labels):
				t_sepsis = np.argmax(labels) - dt_optimal
				best_predictions[max(0, int(t_sepsis + dt_early)) : min(int(t_sepsis + dt_late + 1), int(num_rows))] = 1

			for j in range(n_traj_samples):
				observed_predictions=np.array(predicted[i][j])
				acc_list.append(sum(((best_predictions-observed_predictions)==0).astype(int))/len(best_predictions))
				for i in range(len(observed_predictions)):
					if observed_predictions[i]==1:
						if best_predictions[i]==1:
							truep=truep+1
						else:
							falsep=falsep+1
					else:
						if best_predictions[i]==1:
							falsen=falsen+1
				observed_utilities.append(compute_prediction_utility(labels, observed_predictions))
				best_utilities.append(compute_prediction_utility(labels, best_predictions))
				inaction_utilities.append(compute_prediction_utility(labels, inaction_predictions))

		unnormalized_observed_utility= sum(observed_utilities)
		unnormalized_best_utility= sum(best_utilities)
		unnormalized_inaction_utility= sum(inaction_utilities)
		normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility)


	else:
		acc_list=[]
		observed_utilities=[]
		best_utilities=[]
		inaction_utilities=[]
		truep=0
		falsep=0
		falsen=0
		for i in range(n_batches):
			print("Computing sepsis score... "+ str(i))

			batch_dict = utils.get_next_batch(test_dataloader)

			results  = model.compute_all_losses(batch_dict,
				n_traj_samples = n_traj_samples, kl_coef = kl_coef)


			n_labels = model.n_labels #batch_dict["labels"].size(-1)
			n_traj_samples = results["label_predictions"].size(0)

			predicted=results["label_predictions"]
			truevalue=batch_dict["labels"]
			mask = batch_dict["observed_mask"]
			mask = torch.sum(mask, -1) > 0
			mask = mask.int()
			predicted=predicted>=0
			predicted=predicted.int()
			truevalue=truevalue>=0
			truevalue=truevalue.int()

			predicted=predicted.cpu().numpy()
			truevalue=truevalue.cpu().numpy()


			for i in range(np.shape(truevalue)[0]):
				labels=np.squeeze(truevalue[i])
				num_rows=min(mask[i].argmin(),np.shape(truevalue)[1])
				if (num_rows==0):
					num_rows=np.shape(truevalue)[1]
				labels=labels[:num_rows]
				best_predictions     = np.zeros(num_rows)
				inaction_predictions = np.zeros(num_rows)
				if np.any(labels):
					labels[labels.argmax()]=0
					labels[labels.argmax()]=0
					labels[labels.argmax()]=0
					labels[labels.argmax()]=0
					labels[labels.argmax()]=0
					labels[labels.argmax()]=0
					t_sepsis = np.argmax(labels) - dt_optimal
					best_predictions[max(0, int(t_sepsis + dt_early)) : min(int(t_sepsis + dt_late + 1), int(num_rows))] = 1

				for j in range(n_traj_samples):
					observed_predictions=np.squeeze(predicted[j][i])
					observed_predictions=observed_predictions[:num_rows]
					acc_list.append(sum(((best_predictions-observed_predictions)==0).astype(int))/len(best_predictions))

					for k in range(len(observed_predictions)):
						if observed_predictions[k]==1:
							if best_predictions[k]==1:
								truep=truep+1
							else:
								falsep=falsep+1
						else:
							if best_predictions[k]==1:
								falsen=falsen+1
					observed_utilities.append(compute_prediction_utility(labels, observed_predictions))
					best_utilities.append(compute_prediction_utility(labels, best_predictions))
					inaction_utilities.append(compute_prediction_utility(labels, inaction_predictions))

		unnormalized_observed_utility= sum(observed_utilities)
		unnormalized_best_utility= sum(best_utilities)
		unnormalized_inaction_utility= sum(inaction_utilities)
		normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility)
	acc_list=sum(acc_list)/len(acc_list)
	print(acc_list)
	print(truep/(truep+falsep))
	print(truep/(truep+falsen))
	return normalized_observed_utility
