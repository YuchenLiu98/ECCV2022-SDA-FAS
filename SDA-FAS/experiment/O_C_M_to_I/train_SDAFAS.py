import sys
from numpy.core.defchararray import count
from numpy.lib.utils import source
from sklearn.metrics import roc_auc_score, roc_curve, auc
import copy
from collections import OrderedDict
import json
from torch.serialization import load
sys.path.append('../../')
from utils.utils import sample_frames_direct
from utils.utils import image2cols_batch, col2image_batch
from utils.utils import AverageMeter, Logger, accuracy, mkdirs, adjust_learning_rate, time_to_str
from utils.evaluate import eval
from utils.utils import AverageMeter, accuracy, draw_roc
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate_threshold
from utils.get_loader import get_dataset
from models.DGFAS import embedder_model, Classifier_target
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.optim import create_optimizer
from functools import partial
import random
import numpy as np
from config import config
from datetime import datetime
import time
from timeit import default_timer as timer
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import FASDataset_Aug_both
from torch.autograd import Variable
from torch.nn import functional as F
confidence_threshold = 0.95
def load_model(args):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    checkpoint = torch.load(args.source_model_transformer, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    return model

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    # Optimizer parameters
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--source_model_transformer', default="/home/lyc/SSDG-CVPR2020-master_transformer_imbalance_patch_256_contrastive/experiment/O_C_M_to_I/replay_checkpoint2/resnet18/best_model/model_best_transformer_0.18333_26.pth.tar",
                        help='path for the pretrained source transformer model')
    parser.add_argument('--source_model_embedder_classifier', default="/home/lyc/SSDG-CVPR2020-master_transformer_imbalance_patch_256_contrastive/experiment/O_C_M_to_I/replay_checkpoint2/resnet18/best_model/model_best_0.18333_26.pth.tar",
                        help='path for the pretrained source embedder and classifier model')
    parser.add_argument('--dataset_path', default="/home/liuyuchen/Replay/replaychoose_train_label_four.json",
                        help='dataset path')
    parser.add_argument('--dataset_source_pseudo_path', default="/home/liuyuchen/Replay/Replay_train_pesudo.json",
                        help='path for the source pseudo label for dataset')
    parser.add_argument('--dataset_target_pseudo_path', default="/home/liuyuchen/Replay/Replay_train_pesudo1.json",
                        help='path for the source pseudo label for dataset')
    return parser

def test_and_save(args, train_dataloader, feature_transformer, model):
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    number = 0
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(train_dataloader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            feature_trans = feature_transformer(input)
            cls_out, _ = model(feature_trans, config.norm_flag)
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()
            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                    number += 1
                    if (number % 100 == 0):
                        print('**Testing** ', number, ' photos done!')
    print('**Testing** ', number, ' photos done!')
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
    prob_list_copy = copy.deepcopy(prob_list)
    for i in range(len(prob_list_copy)):
        if prob_list_copy[i] < 0.5:
            prob_list_copy[i] = 1 - prob_list_copy[i]
    # print(len(prob_list_copy))
    count_select = 0
    for i in range(len(prob_list_copy)):
        if prob_list_copy[i] > confidence_threshold:
            count_select = count_select + 1
    print(count_select)

    count_true = 0
    for i in range(len(label_list)):
        if label_list[i]==1 and prob_list[i] > confidence_threshold:
            count_true = count_true + 1
        elif label_list[i]==0 and prob_list[i] <= 1 - confidence_threshold:
            count_true = count_true + 1
    print(count_true/count_select)

    original_all_label_json = json.load(open(args.dataset_path, 'r'))
    f_sample1 = open(args.dataset_source_pseudo_path, 'w')

    final_train_json = []
    for i in range(len(prob_list_copy)):
        for haha in range(0,4):
            dict = {}
            dict['photo_path'] = original_all_label_json[(i)*4+haha]['photo_path']
            dict['GT_label'] = original_all_label_json[(i)*4+haha]['photo_label']
            if prob_list[i] > 0.5:
                dict['photo_label'] = 1
            else:
                dict['photo_label'] = 0
            dict['confidence'] = prob_list_copy[i]
            dict['photo_belong_to_video_ID'] = original_all_label_json[(i)*4+haha]['photo_belong_to_video_ID']
            final_train_json.append(dict)
    json.dump(final_train_json, f_sample1, indent=4)
    f_sample1.close()
    select_prob_list = prob_list
    select_label_list = label_list
    cur_EER_valid, threshold, FRR_list, FAR_list = get_EER_states(select_prob_list, select_label_list)
    ACC_threshold = calculate_threshold(select_prob_list, select_label_list, threshold)
    auc_score = roc_auc_score(select_label_list, select_prob_list)
    draw_roc(FRR_list, FAR_list, auc_score)
    cur_HTER_valid = get_HTER_at_thr(select_prob_list, select_label_list, threshold)
    return [valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, ACC_threshold, threshold]

def test_and_save_target(args, train_dataloader, feature_transformer, model, model_target, log):
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    number = 0
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(train_dataloader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            feature_trans = feature_transformer(input)
            _, feature = model(feature_trans, config.norm_flag)
            cls_out = model_target(feature)
            prob = 1 * F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()
            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                    number += 1
                    if (number % 100 == 0):
                        print('**Testing** ', number, ' photos done!')
    print('**Testing** ', number, ' photos done!')
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
    prob_list_copy = copy.deepcopy(prob_list)
    for i in range(len(prob_list_copy)):
        if prob_list_copy[i] < 0.5:
            prob_list_copy[i] = 1 - prob_list_copy[i]

    count_select = 0
    for i in range(len(prob_list_copy)):
        if prob_list_copy[i] > confidence_threshold:
            count_select = count_select + 1

    count_true = 0
    for i in range(len(label_list)):
        if label_list[i]==1 and prob_list[i] > confidence_threshold:
            count_true = count_true + 1
        elif label_list[i]==0 and prob_list[i] <= 1 - confidence_threshold:
            count_true = count_true + 1
    
    log.write('\n')
    log.write('confidence rate %6.3f' % (count_select/len(prob_list_copy)))
    log.write('\n')
    log.write('label ACC %6.3f' % (count_true/count_select))
    log.write('\n')

    original_all_label_json = json.load(open(args.dataset_path, 'r'))
    source_pseudo_all_label_json = json.load(open(args.dataset_source_pseudo_path, 'r'))
    f_sample1 = open(args.dataset_target_pseudo_path, 'w')
    final_train_json = []
    for i in range(len(prob_list_copy)):
        for haha in range(0,4):
            dict = {}
            dict['photo_path'] = original_all_label_json[(i)*4+haha]['photo_path']
            dict['photo_label'] = source_pseudo_all_label_json[(i)*4+haha]['photo_label']
            dict['confidence'] = source_pseudo_all_label_json[(i)*4+haha]['confidence']
            if prob_list[i] > 0.5:
                dict['photo_label_target'] = 1
            else:
                dict['photo_label_target'] = 0
            dict['confidence_target'] = prob_list_copy[i]
            dict['photo_belong_to_video_ID'] = original_all_label_json[(i)*4+haha]['photo_belong_to_video_ID']
            final_train_json.append(dict)
    json.dump(final_train_json, f_sample1, indent=4)
    f_sample1.close()
    select_prob_list = prob_list
    select_label_list = label_list
    cur_EER_valid, threshold, FRR_list, FAR_list = get_EER_states(select_prob_list, select_label_list)
    ACC_threshold = calculate_threshold(select_prob_list, select_label_list, threshold)
    auc_score = roc_auc_score(select_label_list, select_prob_list)
    draw_roc(FRR_list, FAR_list, auc_score)
    cur_HTER_valid = get_HTER_at_thr(select_prob_list, select_label_list, threshold)
    return (count_select/len(prob_list_copy)), (count_true/count_select)

def loss_CDA_real(feature, protocol_real, protocol_fake, temperature):
    fenzi = torch.exp(torch.div(torch.matmul(feature, protocol_real.t()),temperature))
    fenmu = torch.exp(torch.div(torch.matmul(feature, protocol_real.t()),temperature)) + torch.exp(torch.div(torch.matmul(feature, protocol_fake.t()),temperature))
    result = - torch.log(fenzi/fenmu)
    return result

def loss_CDA_fake(feature, protocol_real, protocol_fake, temperature):
    fenzi = torch.exp(torch.div(torch.matmul(feature, protocol_fake.t()),temperature))
    fenmu = torch.exp(torch.div(torch.matmul(feature, protocol_real.t()),temperature)) + torch.exp(torch.div(torch.matmul(feature, protocol_fake.t()),temperature))
    result = - torch.log(fenzi/fenmu)
    return result

def TSE_loss(teacher_embedding,student_embedding,center):
    teacher_embedding = F.softmax((teacher_embedding - center) / config.temperature, dim=-1)
    teacher_embedding = teacher_embedding.detach()
    return torch.sum(-teacher_embedding * F.log_softmax(student_embedding, dim=-1), dim=-1)
    # return torch.sum(-teacher_embedding * F.log_softmax(student_embedding/config.temperature, dim=-1), dim=-1)

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'

def train(args):
    mkdirs(config.checkpoint_path, config.best_model_path, config.logs)
    # load data
    tgt_train_dataloader_valid, tgt_test_dataloader = get_dataset(
                                       config.tgt_data, config.tgt_test_num_frames, config.batch_size)
    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_ACER = 1.0
    best_model_AUC = 0.0
    confidence_rate = []
    label_ACC = []
    CENTER = torch.zeros([2]).cuda()

    # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:ACER, 5:AUC, 6:threshold
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]
    loss_classifier = AverageMeter()
    loss_dino = AverageMeter()
    loss_classifier_target = AverageMeter()
    loss_classifier_target_contrastive = AverageMeter()
    classifer_top1 = AverageMeter()

    #load pre-trained source transformer encoder model
    feature_transformer = load_model(args)
    feature_transformer = feature_transformer.to(device)

    #fixed source classifier
    classifer_fix = Classifier_target().to(device)

    #load pre-trained source embedder and classifier
    net_s2t = embedder_model().to(device)
    net_s2t_ = torch.load(args.source_model_embedder_classifier)
    net_s2t.load_state_dict(net_s2t_["state_dict"])

    #######initial teacher model 
    teacher_feature_transformer = load_model(args)
    teacher_feature_transformer = teacher_feature_transformer.to(device)

    teacher_net_s2t = embedder_model().to(device)
    teacher_net_s2t_ = torch.load(args.source_model_embedder_classifier)
    teacher_net_s2t.load_state_dict(teacher_net_s2t_["state_dict"])
    ### remove grad
    for k, v in teacher_net_s2t.named_parameters():
        v.requires_grad = False
    for k, v in teacher_feature_transformer.named_parameters():
        v.requires_grad = False

    dict_load  =  OrderedDict()
    dict_load['classifier_layer.weight'] = net_s2t_["state_dict"]['classifier.classifier_layer.weight']
    dict_load['classifier_layer.bias'] = net_s2t_["state_dict"]['classifier.classifier_layer.bias']
    ###use pre-trained weight to initialize the classifier weight
    classifer_fix.load_state_dict(dict_load)

    ###use pre-trained weight as source protocol
    source_classifier_weight = net_s2t_["state_dict"]['classifier.classifier_layer.weight']
    protocol_source_fake = source_classifier_weight[0,:]
    protocol_source_real = source_classifier_weight[1,:]
    protocol_source_fake = torch.div(protocol_source_fake, torch.norm(protocol_source_fake, 2)).cuda()
    protocol_source_real = torch.div(protocol_source_real, torch.norm(protocol_source_real, 2)).cuda()

    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_SDAFAS.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

    ##########use pre-trained source model to generate source oriented pseudo label
    test_args = test_and_save(args, tgt_train_dataloader_valid, feature_transformer, net_s2t)

    log.write('\n=========== Source pseudo label select Test Info===========\n')
    log.write(config.tgt_data, 'Test acc: %5.4f' %(test_args[0]))
    log.write(config.tgt_data, 'Test EER: %5.4f' %(test_args[1]))
    log.write(config.tgt_data, 'Test HTER: %5.4f' %(test_args[2]))
    log.write(config.tgt_data, 'Test AUC: %5.4f' % (test_args[3]))
    log.write(config.tgt_data, 'Test ACC_threshold: %5.4f' % (test_args[4]))
    log.write('\n===============================\n')

    print("Norm_flag: ", config.norm_flag)
    log.write('** start training target model! **\n')
    log.write(
        '--------|------------- VALID -------------|--- classifier ---|------ Current Best ------|--------------|\n')
    log.write(
        '  iter  |   loss   top-1   HTER    AUC    |   loss   top-1   |   top-1   HTER    AUC    |    time      |\n')
    log.write(
        '-------------------------------------------------------------------------------------------------------|\n')
    start = timer()
    criterion = {
        'softmax': nn.CrossEntropyLoss().cuda(),
    }
    optimizer_dict_s2t = [
        {"params": filter(lambda p: p.requires_grad, net_s2t.parameters()), "lr": config.init_lr}
    ]
    #-----------------------optimizer/scheduler------------------------
    optimizer_transformer = create_optimizer(args, feature_transformer)
    optimizer_s2t = optim.SGD(optimizer_dict_s2t, lr=config.init_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    init_param_lr = []
    for param_group in optimizer_s2t.param_groups:
        init_param_lr.append(param_group["lr"])

    iter_per_epoch = 10 

    max_iter = config.max_iter
    epoch = 1
    if(len(config.gpus) > 1):
        classifer_fix = torch.nn.DataParallel(classifer_fix).cuda()

    for iter_num in range(max_iter+1):
        if iter_num % (2*iter_per_epoch)==0:
            current_confidence, current_label_ACC = test_and_save_target(args, tgt_train_dataloader_valid, feature_transformer, net_s2t, classifer_fix, log)
            confidence_rate.append(current_confidence)
            label_ACC.append(current_label_ACC)
            target_train_data = sample_frames_direct(args.dataset_target_pseudo_path)
            target_dataloader = DataLoader(FASDataset_Aug_both(target_train_data, train=True), batch_size=config.batch_size, shuffle=True)
            tgt_train_dataloader_iter_current = iter(target_dataloader)
            tgt_train_iter_per_epoch_current = len(tgt_train_dataloader_iter_current)
        if (iter_num % tgt_train_iter_per_epoch_current == 0):
            tgt_train_dataloader_iter_current = iter(target_dataloader)
        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
        param_lr_tmp = []
        for param_group in optimizer_s2t.param_groups:
            param_lr_tmp.append(param_group["lr"])

        feature_transformer.train(True)
        net_s2t.train(True)

        optimizer_s2t.zero_grad()
        optimizer_transformer.zero_grad()
        if iter_num % iter_per_epoch == 0:
            adjust_learning_rate(optimizer_s2t, epoch, init_param_lr, config.lr_epoch_1, config.lr_epoch_2)

        ######### data prepare #########
        ######### output two augmentation results #########
        tgt_train_img_, tgt_train_img, tgt_train_label, tgt_train_label_target, tgt_train_confidence, tgt_train_confidence_target = tgt_train_dataloader_iter_current.next()

        ######### forward #########
        ######### content destruction for tgt_train_img_ #########
        tgt_train_img_pre = tgt_train_img_.numpy()
        tgt_train_img_pre = tgt_train_img_pre.transpose((0, 2, 3, 1))
        im2col = image2cols_batch(image=tgt_train_img_pre,patch_size=(config.patch,config.patch),stride=config.patch)
        mask_length = im2col.shape[1]
        randox_shunxu_index = np.random.permutation(mask_length)
        current_im2col = im2col[:,randox_shunxu_index,:,:,:]
        tgt_train_img_ = col2image_batch(coldata=current_im2col,imsize=(256, 256),stride=config.patch)
        tgt_train_img_ = tgt_train_img_.transpose((0, 3, 1, 2))
        tgt_train_img_ = torch.from_numpy(tgt_train_img_)
        tgt_train_img_ = tgt_train_img_.float().cuda()

        input_data = tgt_train_img.cuda()
        source_label = tgt_train_label.cuda()
        source_label_target = tgt_train_label_target.cuda()
        source_confidence = tgt_train_confidence.cuda()
        source_confidence_target = tgt_train_confidence_target.cuda()
        
        #####s(x1)#####
        feature_trans = feature_transformer(input_data)
        classifier_label_out, feature = net_s2t(feature_trans, config.norm_flag) 

        #####s(x2)#####
        feature_trans_ = feature_transformer(tgt_train_img_)
        _, feature_ = net_s2t(feature_trans_, config.norm_flag)
        classifier_label_out_ = classifer_fix(feature_)
        #####t(x1)#####
        teacher_feature_trans = teacher_feature_transformer(input_data)
        __, teacher_feature = teacher_net_s2t(teacher_feature_trans, config.norm_flag)
        teacher_classifier_label_out = classifer_fix(teacher_feature)
        #####t(x2)#####
        teacher_feature_trans_ = teacher_feature_transformer(tgt_train_img_)
        ___, teacher_feature_ = teacher_net_s2t(teacher_feature_trans_, config.norm_flag)
        teacher_classifier_label_out_ = classifer_fix(teacher_feature_)

        total_teacher_feature = torch.cat((teacher_classifier_label_out, teacher_classifier_label_out_),0)
        batch_center = torch.sum(total_teacher_feature, dim=0, keepdim=True)
        batch_center = batch_center/(len(total_teacher_feature))
        CENTER = 0.9 * CENTER + 0.1 * batch_center
        
        self_classifier_label_out = classifer_fix(feature)

        alpha = np.float(2.0 / (1.0 + np.exp(-10 * iter_num / float(max_iter//2))) - 1.0)
        ######### cross-entropy loss #########
        cls_loss = 0
        count_number = 0
        total_TSE_loss = 0
        for i in range(input_data.size(0)):
            cls_loss_each = criterion["softmax"](classifier_label_out.narrow(0, i, 1), source_label[i].view(-1))
            if source_confidence[i] >= confidence_threshold:
                cls_loss_each = cls_loss_each 
                count_number = count_number + 1
            else:
                cls_loss_each = 0
            cls_loss = cls_loss + cls_loss_each
            total_TSE_loss = total_TSE_loss + 0.5 * TSE_loss(teacher_classifier_label_out[i],classifier_label_out_[i],CENTER) + 0.5 * TSE_loss(teacher_classifier_label_out_[i],self_classifier_label_out[i],CENTER)
        if count_number == 0:
            cls_loss = 0
        else:
            cls_loss = cls_loss/count_number
        total_TSE_loss = total_TSE_loss/input_data.size(0)

        cls_loss_target = 0
        cda_loss = 0
        count_number_target = 0
        for i in range(input_data.size(0)):
            cls_loss_each_target = criterion["softmax"](self_classifier_label_out.narrow(0, i, 1), source_label_target[i].view(-1))
            if source_confidence_target[i] >= confidence_threshold:
                cls_loss_each_target = cls_loss_each_target #* source_confidence[i]
                count_number_target = count_number_target + 1
                if source_label_target[i] == 1:
                    cda_loss_each_target = loss_CDA_real(feature[i], protocol_source_real, protocol_source_fake, config.temperature)
                else:
                    cda_loss_each_target = loss_CDA_fake(feature[i], protocol_source_real, protocol_source_fake, config.temperature)
            else:
                cls_loss_each_target = 0
                cda_loss_each_target = 0
            cls_loss_target = cls_loss_target + cls_loss_each_target
            cda_loss = cda_loss + cda_loss_each_target
        if count_number_target == 0:
            cls_loss_target = 0
            cda_loss = 0
        else:
            cls_loss_target = cls_loss_target/count_number_target
            cda_loss = cda_loss/count_number_target

        ######### backward #########config.lambda_triplet * triplet 
        total_loss = cls_loss * (1-alpha) + alpha * cls_loss_target + 0.3*cda_loss + 0.15*total_TSE_loss
        total_loss.backward()
        optimizer_transformer.step()
        optimizer_transformer.zero_grad()
        optimizer_s2t.step()
        optimizer_s2t.zero_grad()
        
        m=0.999
        for param_q, param_k in zip(feature_transformer.parameters(), teacher_feature_transformer.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        
        for param_q, param_k in zip(net_s2t.parameters(), teacher_net_s2t.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


        loss_classifier.update((1-alpha) * cls_loss.item())
        loss_classifier_target.update(alpha * cls_loss_target.item())
        loss_classifier_target_contrastive.update(cda_loss.item())
        loss_dino.update(total_TSE_loss.item())
        
        acc = accuracy(classifier_label_out.narrow(0, 0, input_data.size(0)), source_label, topk=(1,))
        classifer_top1.update(acc[0])
        print('\r', end='', flush=True)
        print(
            '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s'
            % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'))
            , end='', flush=True)

        if (iter_num != 0 and (iter_num+1) % iter_per_epoch == 0):
            # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold, 6:ACC_threshold
            valid_args = eval(tgt_test_dataloader, feature_transformer, net_s2t, classifer_fix, config.norm_flag)
            # judge model according to HTER
            is_best = valid_args[3] <= best_model_HTER
            best_model_HTER = min(valid_args[3], best_model_HTER)
            threshold = valid_args[5]
            if (valid_args[3] <= best_model_HTER):
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]

            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER, threshold]
            # save_checkpoint(save_list, is_best, net, config.gpus, config.checkpoint_path, config.best_model_path)
            # save_checkpoint(save_list, is_best, net_s2t, config.gpus, config.checkpoint_path, config.best_model_path)
            # save_checkpoint_transformer(save_list, is_best, feature_transformer, config.gpus, config.checkpoint_path, config.best_model_path)
            print('\r', end='', flush=True)
            log.write(
                '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s   %s'
                % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'),
                param_lr_tmp[0]))
            log.write('\n')
            # log.write('cls loss %6.3f, triplet loss %6.3f, clss_patch loss %6.3f' % (loss_classifier.avg,loss_triplet_patch.avg,loss_classifier_patch.avg))
            # log.write('cls loss %6.3f, triplet loss %6.3f, clss_patch loss %6.3f' % (loss_classifier.avg,0,loss_classifier_patch.avg))
            if count_number == 0:
                log.write('cls loss %6.3f, target cls loss %6.3f, target contrastive loss %6.3f, dino loss %6.3f' % (0, loss_classifier_target.avg, 0, loss_dino.avg))
            else:
                log.write('cls loss %6.3f, target cls loss %6.3f, target contrastive loss %6.3f, dino loss %6.3f' % (loss_classifier.avg, loss_classifier_target.avg, loss_classifier_target_contrastive.avg, loss_dino.avg))
            
            log.write('\n')
            time.sleep(0.01)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    train(args)









































