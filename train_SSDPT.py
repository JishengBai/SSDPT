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
        
        # if idx == 0:
        #     data = np.zeros((len(file_list), vectors.shape[0], vectors.shape[1]), float)
        # data[idx, :] = vectors

    return data

########################################################################
    
def train_one_epoch(model, loss_f1, loss_f2, data_loader, optimizer, epoch, device,
                    is_mixup, spec_aug, is_fmix, lr_scheduler, loss_weight):
    model.train()
    start = time.time()
    
    loss_numpy, loss_mse_numpy, loss_ce_numpy, correct_numpy, train_batch_num = 0, 0, 0, 0, 0
    
    for idx, (feature, target) in enumerate(data_loader):
        
        feature = feature.reshape(feature.size(0), 1, -1, feature.size(2))
        feature = feature.to(device)
        target = target.to(device)

        if is_mixup:
            lam = np.random.beta(1,1)
            index = torch.randperm(feature.size(0)).to(device)
            feature = lam*feature + (1-lam)*feature[index,:]
            target_a, target_b = target, target[index]
            x_hat, output = model(feature, spec_aug)
            # output = model(feature) 
            loss_mse = loss_f1(x_hat, feature.view(feature.size(0), -1, feature.size(3)))
            loss_ce = lam * loss_f2(output, target_a) + (1 - lam) * loss_f2(output, target_b)
            loss = loss_weight[0]*loss_mse+loss_weight[1]*loss_ce

        else:
            
            # output = model(feature) 
            x_hat, output = model(feature, spec_aug)
            loss_mse = loss_f1(x_hat, feature.view(feature.size(0), -1, feature.size(3)))

            loss_ce = loss_f2(output, target)
            loss = loss_weight[0]*loss_mse+loss_weight[1]*loss_ce
        
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # lr_scheduler.step_update(epoch * num_steps + idx)
        
        softmax_output = torch.softmax(output, dim=1)
        prediction = torch.max(softmax_output, 1)[1] 
        correct = (prediction == target).sum().float()
        correct = torch.div(correct, target.size(0))
        
        loss_mse_numpy += loss_mse.detach().cpu().numpy()
        loss_ce_numpy += loss_ce.detach().cpu().numpy()
        loss_numpy += loss.detach().cpu().numpy()
        correct_numpy += correct.detach().cpu().numpy()
        train_batch_num += 1 

    lr_scheduler.step()
    
    optimizer.zero_grad()
    
    loss_numpy = loss_numpy/train_batch_num
    loss_mse_numpy = loss_mse_numpy/train_batch_num
    loss_ce_numpy = loss_ce_numpy/train_batch_num
    correct_numpy = correct_numpy/train_batch_num
     
    torch.cuda.synchronize()

    end = time.time() 
    
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    epoch_time = end - start
    print("   TRAIN:   ")
    print(
    "| time: %.2f" % epoch_time,
    "| loss: %.4f" % loss_numpy,
    "| loss mse: %.4f" % loss_mse_numpy,
    "| loss ce: %.4f" % loss_ce_numpy,
    "| acc: %.4f" % correct_numpy,
    "| memory: %.0f MB" % memory_used
        )

def pred_step(model, feature, loss_f1):
    
    model.eval()

    with torch.no_grad():

        
        x_hat, output = model(feature, False)
        output = torch.softmax(output, dim=1)
        loss_mse = loss_f1(x_hat, feature.view(feature.size(0), -1, feature.size(3)))
        
    return loss_mse, output

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

def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

########################################################################

########################################################################
if __name__ == "__main__":
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    # mode = True
    mode = com.command_line_chk()
    # mode = True
    if mode is None:
        sys.exit(-1)
        
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)
    log_file_path = os.path.join(param["model_directory"], 'terminal_log.log')
    sys.stdout = Logger(log_file_path)
    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    os.makedirs(param["result_directory"], exist_ok=True)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    performance_over_all = []

    best_score_dict = {}
    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
    # [ToyCar, ToyTrain, fan, gearbox, pump, slider, valve]
        
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx+1, total=len(dirs)))
            
        # set path
        machine_type = os.path.split(target_dir)[1]
        
        model_file_path = "{model}/model_{machine_type}.pth".format(model=param["model_directory"],
                                                                      machine_type=machine_type)
        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue

        # pickle file for storing section names
        section_names_file_path = "{model}/section_names_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                    machine_type=machine_type)
        # pickle file for storing anomaly score distribution
        score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                machine_type=machine_type)
         #######################################################################
        print("============== PARAMS LOADING ==============")
        parameter_dict = load_pars()
        np.save(os.path.join(
                        param["model_directory"], str(machine_type)+'_parameter_dict.npy'), 
                                                        parameter_dict, allow_pickle=True)
        ### train
        degrade_step = parameter_dict['degrade_step']
        is_mixup = parameter_dict['is_mixup']
        spec_aug = parameter_dict['spec_aug']
        is_fmix = parameter_dict['is_fmix']
        
        optimizer_eps = parameter_dict['optimizer_eps']
        optimizer_betas = parameter_dict['optimizer_betas']
        weight_decay = parameter_dict['weight_decay']
        ### CNN domian  
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
        if os.path.exists(model_file_path) and not load_ckpt:
            com.logger.info("model exists")
            continue
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
        train_feature_tensor = torch.from_numpy(train_data).float()
        train_target_tensor = torch.from_numpy(train_label).long()
        train_dataset = TensorDataset(train_feature_tensor, train_target_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=param["fit"]["batch_size"], 
                                  shuffle=True, num_workers=2, pin_memory=True)
        #######################################################################
        print("============== TEST DATA GENERATING ==============")
        test_data_dict = gen_test_data(param, target_dir, data_reshape)
        ############################################################################
        #######################################################################
        print("============== MODEL TRAINING ==============")

        model = SSDPT(param["feature"]["n_frames"], param["feature"]["n_mels"],
                                        n_sections,
                                        nhead, dim_feedforward, n_layers, dropout)
       
        # optimizer = optim.Adam(model.parameters(), lr=param["fit"]["lr"], 
        #                         betas=optimizer_betas, eps=optimizer_eps, weight_decay=weight_decay)
        optimizer = optim.AdamW(model.parameters(), lr=param["fit"]["lr"])  
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
 
        device = torch.device('cuda')
        
        criterion_1 = nn.MSELoss()
        criterion_2 = nn.CrossEntropyLoss()
        model.to(device)
        
        if load_ckpt:
            print("============== LOADING CHECKPOINT ==============")
            checkpoint = torch.load(model_file_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        else:
            start_epoch = 0
            
        ############################################################################
        total_parameters = sum([param.nelement() for param in model.parameters()])
        print("Number of parameters: ", total_parameters)
        
        test_score_list = []
        best_epoch = 0
        best_score = 0
        
        for epoch in range(start_epoch, param["fit"]["epochs"], 1):
            print("============== EPOCH {} ==============".format(epoch))
            train_one_epoch(model, criterion_1, criterion_2,
                            train_loader, optimizer, epoch, device, is_mixup, spec_aug, is_fmix, scheduler,
                            loss_weight)
            
            # calculate y_pred for fitting anomaly score distribution
            pred_re_loss = []
            pred_logsoft = []
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
                    pred_re_loss.append(re_loss)
                    logsoft = np.mean(np.log(np.maximum(1.0 - p, sys.float_info.epsilon) 
                                          - np.log(np.maximum(p, sys.float_info.epsilon))))
                    pred_logsoft.append(logsoft)
                    anomaly_score = score_weight[0]*re_loss + score_weight[1]*logsoft
                    y_pred.append(anomaly_score)
                    start_idx += n_vectors_ea_file
                    
            print("mse loss:", np.mean(np.asarray(pred_re_loss)),
                  "logsoft:", np.mean(np.asarray(pred_logsoft)),
                  "anomaly score:", np.mean(np.asarray(y_pred))
                    )
            # fit anomaly score distribution
            shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred)
    
            if epoch % degrade_step == 0:
                print("   TEST:   ")

                csv_lines.append([machine_type])
                csv_lines.append(["section", "domain", "AUC", "pAUC", "precision", "recall", "F1 score"])
                performance = []

                # determine threshold for decision
                # decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)
  
                trained_section_names = joblib.load(section_names_file_path)
                n_sections = trained_section_names.shape[0]
  
                t0 = time.time()
                
                dir_names = ["source_test", "target_test"]
    
                for dir_name in dir_names:
        
                    #list machine id
                    section_names = com.get_section_names(target_dir, dir_name=dir_name)
                    
                    # print(dir_name)
                    
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

                        # setup anomaly score file path
                        anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{section_name}_{dir_name}.csv".format(result=param["result_directory"],
                                                                                                                          machine_type=machine_type,
                                                                                                                          section_name=section_name,
                                                                                                                          dir_name=dir_name)
                        anomaly_score_list = []
        
                        # setup decision result file path
                        decision_result_csv = "{result}/decision_result_{machine_type}_{section_name}_{dir_name}.csv".format(result=param["result_directory"],
                                                                                                                              machine_type=machine_type,
                                                                                                                              section_name=section_name,
                                                                                                                              dir_name=dir_name)
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
                            test_anomaly_score = score_weight[0]*re_loss + score_weight[1]*logsoft
                            y_pred[file_idx] = test_anomaly_score

                            # store anomaly scores
                            anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
        
                            # store decision results
                            # if y_pred[file_idx] > decision_threshold:
                            #     decision_result_list.append([os.path.basename(file_path), 1])
                            # else:
                            #     decision_result_list.append([os.path.basename(file_path), 0])
                           
                        # print(section_name) 
                        # print("mse loss:", np.mean(np.asarray(test_re_loss)),
                        #   "logsoft:", np.mean(np.asarray(test_logsoft)),
                        #   "anomaly score:", np.mean(np.asarray(y_pred))
                        #   )

                        # output anomaly scores
                        # save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
                        # com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))
        
                        # output decision results
                        # save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
                        # com.logger.info("decision result ->  {}".format(decision_result_csv))
        
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
            
                test_score_list.append(hmean_performance[0])
                print('| Test score:', round(hmean_performance[0], 4))
            
                if hmean_performance[0]>=best_score:
                    # gamma_params = [shape_hat, loc_hat, scale_hat]
                    # joblib.dump(gamma_params, score_distr_file_path)
                    best_epoch = epoch
                    best_score = hmean_performance[0]
                    best_score_dict[machine_type] = [best_epoch, list(amean_performance), 
                                                      list(hmean_performance)]    
                    checkpoint = {
                        "epoch": epoch,
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict()}     
                    torch.save(checkpoint, model_file_path)
                    print('Model saved to {}'.format(model_file_path))
            print('| Best_epoch:', best_epoch, '| best_score', round(best_score, 4))
            ############################################################################    
        torch.cuda.empty_cache()
        del train_data
        del train_label
        del test_data_dict
        del train_loader
        del model
        del optimizer
        del scheduler
        gc.collect()
        print("============== END TRAINING ==============")  
        ############################################################################
    best_AUC_score_list = []
    best_pAUC_score_list = []
    for mac in best_score_dict:
        print('Machine type:', mac, 
              'best AUC:', round(best_score_dict[mac][2][0],4), 
              'best pAUC:', round(best_score_dict[mac][2][1],4),
              'best epoch:', round(best_score_dict[mac][0],4)
              )
        best_AUC_score_list.append(best_score_dict[mac][2][0])
        best_pAUC_score_list.append(best_score_dict[mac][2][1])
    best_AUC_score_list = np.asarray(best_AUC_score_list)
    best_pAUC_score_list = np.asarray(best_pAUC_score_list)
    print('Ave best AUC score:', np.mean(best_AUC_score_list),
          'Ave best pAUC score:', np.mean(best_pAUC_score_list),)
    best_score_dict_path = os.path.join(param["model_directory"], 'best_score_dict.npy')
    np.save(best_score_dict_path, best_score_dict, allow_pickle=True) 

