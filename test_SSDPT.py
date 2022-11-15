#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################
# import default libraries
########################################################################
import os
import sys
import gc
import csv
import numpy as np
import scipy.stats
# from import
from tqdm import tqdm
try:
    from sklearn.externals import joblib
except:
    import joblib
# original lib
import common as com
########################################################################
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
import os
from sklearn import metrics
from load_parameters import load_pars
from torch.utils.data import DataLoader, TensorDataset
from SSDPT import DPTrans, SSDPT
from torchsummary import summary
from torchstat import stat
########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

########################################################################
# get data from the list for file paths
########################################################################
def file_list_to_data(file_list,
                      msg="calc...",
                      n_mels=64,
                      n_frames=5,
                      n_hop_frames=1,
                      n_fft=1024,
                      hop_length=512,
                      power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vectors = com.file_to_vectors(file_list[idx],
                                                n_mels=n_mels,
                                                n_frames=n_frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        vectors = vectors[: : n_hop_frames, :]
        if idx == 0:
            data = np.zeros((len(file_list) * vectors.shape[0], dims), float)
        data[vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :] = vectors

    return data

def gen_data(param, target_dir, section_names_file_path, reshape=False):
    section_names = com.get_section_names(target_dir, dir_name="train")
    unique_section_names = np.unique(section_names)
    n_sections = unique_section_names.shape[0]
    joblib.dump(unique_section_names, section_names_file_path)
    n_files_ea_section = []  
    # data = np.empty((0, param["feature"]["n_frames"] * param["feature"]["n_mels"]), float)
    data = []
    for section_idx, section_name in enumerate(unique_section_names):
        files, y_true = com.file_list_generator(target_dir=target_dir,
                                                section_name=section_name,
                                                dir_name="train",
                                                mode=mode)

        n_files_ea_section.append(len(files))

        data_ea_section = file_list_to_data(files,
                                            msg="generate train_dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            n_frames=param["feature"]["n_frames"],
                                            n_hop_frames=param["feature"]["n_hop_frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"])
        
        # data = np.append(data, data_ea_section, axis=0)
        data.extend(data_ea_section)
    data = np.asarray(data)
    n_all_files = sum(n_files_ea_section)
    n_vectors_ea_file = int(data.shape[0] / n_all_files)
    condition = np.zeros(data.shape[0], float)
    start_idx = 0
    for section_idx in range(n_sections): # section_idx 0, 1, 2
        n_vectors = n_vectors_ea_file * n_files_ea_section[section_idx]
        condition[start_idx : start_idx + n_vectors] = section_idx
        start_idx += n_vectors

    if reshape:
        data = data.reshape(n_all_files, -1)
        n_vectors_ea_file = int(data.shape[0] / n_all_files)
        condition = np.zeros(data.shape[0], float)
        start_idx = 0
        for section_idx in range(n_sections): # section_idx 0, 1, 2
            n_vectors = n_vectors_ea_file * n_files_ea_section[section_idx]
            condition[start_idx : start_idx + n_vectors] = section_idx
            start_idx += n_vectors

    train_data_dict = {}
    
    train_data_dict['train_data'] = data
    train_data_dict['train_label'] = condition

    return train_data_dict, n_files_ea_section, n_vectors_ea_file, n_sections

def gen_test_data(param, target_dir, reshape=False):
    test_data_dict = {}
    dir_names = ["source_test", "target_test"]
    for dir_name in dir_names:
        #list machine id
        section_names = com.get_section_names(target_dir, dir_name=dir_name)            
        test_data_dict[dir_name] = {}
        for section_name in section_names:
            # load test file
            files, y_true = com.file_list_generator(target_dir=target_dir,
                                        section_name=section_name,
                                        dir_name=dir_name,
                                        mode=mode)
            test_data_dict[dir_name][section_name] = {}
            for file_idx, file_path in enumerate(files):
                try:
                    data = com.file_to_vectors(file_path,
                                                    n_mels=param["feature"]["n_mels"],
                                                    n_frames=param["feature"]["n_frames"],
                                                    n_fft=param["feature"]["n_fft"],
                                                    hop_length=param["feature"]["hop_length"],
                                                    power=param["feature"]["power"])
                except:
                    com.logger.error("File broken!!: {}".format(file_path))
                    
                test_data_dict[dir_name][section_name][file_idx] = data 
    return test_data_dict


def pred_step(model, feature, loss_f1):
    
    model.eval()

    with torch.no_grad():

        # output = model(feature) 
        x_hat, output = model(feature, False)
        output = torch.softmax(output, dim=1)
        loss_mse = loss_f1(x_hat, feature.view(feature.size(0), -1, feature.size(3)))
        
    return loss_mse, output

def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

########################################################################

########################################################################
if __name__ == "__main__":
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()

    if mode is None:
        sys.exit(-1)
        
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)
    log_file_path = os.path.join(param["model_directory"], 'test_log.log')
    sys.stdout = Logger(log_file_path)
    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    os.makedirs(param["result_directory"], exist_ok=True)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    performance_over_all = []

    h_AUC_arr = np.zeros((7, 10))
    h_pAUC_arr = np.zeros((7, 10))

    best_score_dict = {}
    # loop of the base directory

    for idx, target_dir in enumerate(dirs):
    # [ToyCar, ToyTrain, fan, gearbox, pump, slider, valve]
    # target_dir = dirs[idx]
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx+1, total=len(dirs)))
            
        # set path
        machine_type = os.path.split(target_dir)[1]
        
        best_score_dict[machine_type] = {}
        
        model_file_path = "{model}/model_{machine_type}.pth".format(model=param["model_directory"],
                                                                      machine_type=machine_type)
        # pickle file for storing section names
        section_names_file_path = "{model}/section_names_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                    machine_type=machine_type)
        # pickle file for storing anomaly score distribution
        score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                machine_type=machine_type)
         #######################################################################
        print("============== PARAMS LOADING ==============")
        parameter_dict = load_pars()

        ### transformer_encoder 
        nhead = parameter_dict['nhead']
        dim_feedforward = parameter_dict['dim_feedforward']
        n_layers = parameter_dict['n_layers']  
        dropout = parameter_dict['dropout']
        
        data_reshape = parameter_dict['reshape']
        load_ckpt = parameter_dict['load_ckpt']
        
        loss_weight = parameter_dict['loss_weight']
        score_weight = parameter_dict['score_weight']
        #######################################################################
        print("============== TRAIN DATA GENERATING ==============")
        train_data_dict, n_files_ea_section, n_vectors_ea_file, n_sections = gen_data(
                            param, target_dir, section_names_file_path, data_reshape)
        train_data = train_data_dict['train_data']
        train_label = train_data_dict['train_label']
        train_data = train_data.reshape(train_data.shape[0], -1, 
                                        param["feature"]["n_mels"])
        # #data shape   (96288, 64, 128, 1) 
        train_num, frames, bins = train_data.shape
        print("train num:", train_num, "frames:", frames, "bins:", bins, 'n_vectors:', n_vectors_ea_file)
        # train_feature_tensor = torch.from_numpy(train_data).float()
        # train_target_tensor = torch.from_numpy(train_label).long()
        # train_dataset = TensorDataset(train_feature_tensor, train_target_tensor)
        # train_loader = DataLoader(dataset=train_dataset, batch_size=param["fit"]["batch_size"], 
        #                           shuffle=True, num_workers=2, pin_memory=True)
        #######################################################################
        print("============== TEST DATA GENERATING ==============")
        test_data_dict = gen_test_data(param, target_dir, data_reshape)
        ############################################################################
        #######################################################################
        print("============== MODEL TRAINING ==============")
        
        model = SSDPT(param["feature"]["n_frames"], param["feature"]["n_mels"],
                                        n_sections,
                                        nhead, dim_feedforward, n_layers, dropout)
    
        device = torch.device('cuda')
        
        criterion_1 = nn.MSELoss()
        criterion_2 = nn.CrossEntropyLoss()
        
        model.to(device)
        
        print("============== LOADING CHECKPOINT ==============")
        checkpoint = torch.load(model_file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
            
        ############################################################################
        # stat(model.to('cpu'), (1, 64, 128))
        
        # summary(model, input_size=(1, param["feature"]["n_frames"], param["feature"]["n_mels"]))
        
        y_pred = []
        start_idx = 0
        for section_idx in range(n_sections):
            for file_idx in range(n_files_ea_section[section_idx]):
                pred_feature_batch = train_data[start_idx : start_idx + n_vectors_ea_file, : ]
                pred_feature_batch = pred_feature_batch.reshape(pred_feature_batch.shape[0], 1,-1, param["feature"]["n_mels"])
                pred_feature_batch = torch.from_numpy(pred_feature_batch).float()
                pred_feature_batch = pred_feature_batch.to(device)
                re_loss, p = pred_step(model, pred_feature_batch, criterion_1)
                p = p.detach().cpu().numpy()[:, section_idx : section_idx + 1]
                re_loss = re_loss.detach().cpu().numpy()
                # pred_re_loss.append(re_loss)
                logsoft = np.mean(np.log(np.maximum(1.0 - p, sys.float_info.epsilon) 
                                      - np.log(np.maximum(p, sys.float_info.epsilon))))
                # pred_logsoft.append(logsoft)
                anomaly_score = 0.001*re_loss + score_weight[1]*logsoft
                y_pred.append(anomaly_score)
                start_idx += n_vectors_ea_file

        # fit anomaly score distribution
        shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred)
        decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)
        print('decision_threshold', decision_threshold)
        # total_parameters = sum([param.nelement() for param in model.parameters()])
        # print("Number of parameters: ", total_parameters)
        
        # best_weight = 0
        # best_score = 0
        
        weight_list = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005]
        

        
        for weight_idx, weight in enumerate(weight_list):

            print("   TEST:   ")

            csv_lines.append([machine_type])
            csv_lines.append(["section", "domain", "AUC", "pAUC", "precision", "recall", "F1 score"])
            performance = []

            trained_section_names = joblib.load(section_names_file_path)
            n_sections = trained_section_names.shape[0]
  
            t0 = time.time()
            
            dir_names = ["source_test", "target_test"]

            # y_prd_list = []
            # y_true_list = []
            
            for dir_name in dir_names:
    
                #list machine id
                section_names = com.get_section_names(target_dir, dir_name=dir_name)
                
                for section_name in section_names:

                    temp_array = np.nonzero(trained_section_names == section_name)[0]
                    if temp_array.shape[0] == 0:
                        section_idx = -1
                    else:
                        section_idx = temp_array[0] 
    
                    # load test file
                    files, y_true = com.file_list_generator(target_dir=target_dir,
                                                section_name=section_name,
                                                dir_name=dir_name,
                                                mode=mode)

                    decision_result_list = []
    
                    # print("\n============== BEGIN TEST FOR A SECTION ==============")
                    test_re_loss = []
                    test_logsoft = []
                    y_pred = [0. for k in files]
                    for file_idx, file_path in enumerate(files):
                        test_data = test_data_dict[dir_name][section_name][file_idx] 
                        test_data = test_data.reshape(test_data.shape[0], 1, -1, param["feature"]["n_mels"])                
                        test_data = torch.from_numpy(test_data).float()
                        test_data = test_data.to(device)
                        re_loss, p = pred_step(model, test_data, criterion_1)
                        p = p.detach().cpu().numpy()[:, section_idx : section_idx + 1]
                        
                        re_loss = re_loss.detach().cpu().numpy()
                        test_re_loss.append(re_loss)
                        logsoft = np.mean(np.log(np.maximum(1.0 - p, sys.float_info.epsilon) 
                                              - np.log(np.maximum(p, sys.float_info.epsilon))))
                        test_logsoft.append(logsoft)
                        
                        test_anomaly_score = weight*re_loss + score_weight[1]*logsoft
                        y_pred[file_idx] = test_anomaly_score
        
                    # append AUC and pAUC to lists
                    auc = metrics.roc_auc_score(y_true, y_pred)
                    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                    # tn, fp, fn, tp = metrics.confusion_matrix(y_true, [1 if x > decision_threshold else 0 for x in y_pred]).ravel()
                    # prec = tp / np.maximum(tp + fp, sys.float_info.epsilon)
                    # recall = tp / np.maximum(tp + fn, sys.float_info.epsilon)
                    # f1 = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)
                    prec, recall, f1 = 0, 0, 0
                    csv_lines.append([section_name.split("_", 1)[1], dir_name.split("_", 1)[0], auc, p_auc, prec, recall, f1])
                    performance.append([auc, p_auc, prec, recall, f1])
                    performance_over_all.append([auc, p_auc, prec, recall, f1])

                # calculate averages for AUCs and pAUCs
            amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean", ""] + list(amean_performance))
            hmean_performance = scipy.stats.hmean(np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0)
            csv_lines.append(["harmonic mean", ""] + list(hmean_performance))
            csv_lines.append([])

            print('weight:', weight, 'h_AUC', round(hmean_performance[0], 4), 'p_AUC', round(hmean_performance[1], 4))
            
            h_AUC_arr[idx, weight_idx] =  hmean_performance[0]
            h_pAUC_arr[idx, weight_idx] = hmean_performance[1]
                
        h_AUC_performance = scipy.stats.hmean(h_AUC_arr, axis=0).reshape(1, -1)
        h_pAUC_performance = scipy.stats.hmean(h_pAUC_arr, axis=0).reshape(1, -1)
        
        h_AUC_arr = np.concatenate((h_AUC_arr, h_AUC_performance), axis=0)
        h_pAUC_arr = np.concatenate((h_pAUC_arr, h_pAUC_performance), axis=0)
        h_AUC_beta_dict = {}
        h_AUC_beta_dict['h_AUC'] = h_AUC_arr
        h_AUC_beta_dict['h_pAUC'] = h_pAUC_arr
        np.save(os.path.join(param["model_directory"], 'h_AUC_beta_dict.npy'), h_AUC_beta_dict, allow_pickle=True)
        
        if hmean_performance[0]>=best_score:
            best_weight = weight
            best_score = hmean_performance[0]
        best_score_dict[machine_type][best_weight] = best_score        
        print('best_weight:', best_weight, '| best_score', round(best_score, 4))    
            ############################################################################    
        torch.cuda.empty_cache()
        
        del test_data_dict
        del model

        gc.collect()
        print("============== END TESTING ==============")  
        # ############################################################################

