class DefaultConfigs(object):
    seed = 666
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.0001
    lr_epoch_1 = 0
    lr_epoch_2 = 100
    gpus = "1"
    patch = 64
    batch_size = 24
    norm_flag = True
    max_iter = 600
    temperature = 0.1
    # source data information
    src1_data = 'CASIA_label'
    src1_train_num_frames = 1
    src2_data = 'OULU'
    src2_train_num_frames = 1
    src3_data = 'MSU'
    src3_train_num_frames = 1
    # target data information
    tgt_data = 'replay'
    tgt_test_num_frames = 2
    # paths information
    checkpoint_path = './' + tgt_data + '_checkpoint_SDAFAS/'
    best_model_path = './' + tgt_data + '_best_model_SDAFAS/'
    logs = './logs_SDAFAS/'

config = DefaultConfigs()
