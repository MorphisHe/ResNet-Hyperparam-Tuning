import os
from matplotlib import pyplot as plt
 


def draw_plot(names=[], acc_list=[], output_dir="", title="", output_name="", xlim=[]):
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize =(19,11))
    ax.set_xlim(xlim)
 
    # Horizontal Bar Plot
    ax.barh(names, acc_list)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)

    # Add x, y gridlines
    ax.grid(b = True, color ='black',
            linestyle ='-.', linewidth = 1.0,
            alpha = 0)
 
    # Show top values
    ax.invert_yaxis()
 
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                str(round((i.get_width()), 2)),
                fontsize = 20, fontweight ='bold',
                color ='black')
 
    # Add Plot Title
    ax.set_title(title,
                loc ='left', fontsize=30)
    
    # Show Plot
    plt.savefig(os.path.join(output_dir, output_name), bbox_inches='tight')


def extra_data_from_log(src_dir="", dir_names=[], output_dir="", names=[], title="", output_name="", baseline_setting="", xlim=[]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    acc_list = []
    best_epoch_num = []
    for dir_name in dir_names:
        print(dir_name)
        dir_name = os.path.join(src_dir, dir_name)

        # search for log file
        fns = os.listdir(dir_name)
        log_fn = None
        for fn in fns:
            if "log" in fn:
                log_fn = fn
                break
        
        # read log file lines
        lines = open(os.path.join(dir_name, log_fn)).readlines()

        test_loss, test_acc = 0, 0
        best_model_epoch = 0
        epoch_number = 0
        for line in lines:
            if "Eval Devset" in line:
                # get epoch number
                pieces = line.split("Epoch #")
                epoch_number = int(pieces[1].split(":")[0])
            elif "Test Devset" in line:
                # Sample from log file:   Epoch #13: Average Loss 0.66763 - Epoch Acc: 76.49000 - Epoch Testing Time: 0.035 min(s)
                pieces = line.split("-")
                test_loss = float(pieces[0].split("Loss")[-1].strip())
                test_acc = float(pieces[1].split("Acc:")[-1].strip())
            elif "Saving new best-model" in line:
                best_model_epoch = epoch_number

        acc_list.append(test_acc)
        best_epoch_num.append(best_model_epoch)
    
    new_names = []
    for idx, (name, best_epoch) in enumerate(zip(names, best_epoch_num)):
        if idx == 0:
            #new_names.append(f"{name}\n{baseline_setting}\nepochs:{best_epoch}")
            new_names.append(f"{name}\n{baseline_setting}")
        else:
            #new_names.append(f"{name}\nepochs:{best_epoch}")
            new_names.append(f"{name}")
    
    draw_plot(names=new_names, acc_list=acc_list, output_dir=output_dir, title=title, output_name=output_name, xlim=xlim)


if __name__ == "__main__":
    output_dir = "output_plots/bar_plots"

    # data_aug
    src_dir = "outputs/data_aug/all"
    names = ["baseline", "norm", "norm+crop", "norm+crop+flip", "norm+crop+flip+rotate(5deg)"]
    dir_names = ["baseline", "baseline_norm", "baseline_norm_crop", "baseline_norm_crop_flip", "baseline_norm_crop_flip_rotate5"]
    title = "Testset Accuracy"
    output_name = "data_aug_bar.jpg"
    baseline_setting = "no augs"
    xlim = [75, 82]
    extra_data_from_log(src_dir, dir_names, output_dir, names, title, output_name, baseline_setting, xlim)

    # wd + lr
    src_dir = "outputs/lr_weightDecay/all"
    names = ["baseline+augs", "wd:0.01+lr:0.01", "wd:3e-3+lr:0.01", "wd:5e-4+lr:0.01", "wd:3e-3+lr:0.001", "wd:5e-4+lr:0.001"]
    dir_names = ["baseline_norm_crop_flip_rotate5", "baseline_norm_crop_flip_rotate5_wd0.01", 
                 "baseline_norm_crop_flip_rotate5_wd0.003", "baseline_norm_crop_flip_rotate5_wd5e-4", 
                 "baseline_norm_crop_flip_rotate5_wd0.003_lr0.001", "baseline_norm_crop_flip_rotate5_wd5e-4_lr0.001"]
    title = "Testset Accuracy"
    output_name = "lr_wd_bar.jpg"
    baseline_setting = "wd:0.1+lr:0.01"
    xlim = [78, 92]
    extra_data_from_log(src_dir, dir_names, output_dir, names, title, output_name, baseline_setting, xlim)
    
    # batch size
    src_dir = "outputs/batch_size/all"
    names = ["baseline+augs+wd+lr", "train batch:32", "train batch:64", "train batch:256"]
    dir_names = ["baseline_norm_crop_flip_rotate5_wd5e-4", "baseline_norm_crop_flip_rotate5_wd5e-4_bs32",
                 "baseline_norm_crop_flip_rotate5_wd5e-4_bs64", "baseline_norm_crop_flip_rotate5_wd5e-4_bs256"]
    title = "Testset Accuracy"
    output_name = "bs_bar.jpg"
    baseline_setting = "train batch:128"
    xlim = [88, 92]
    extra_data_from_log(src_dir, dir_names, output_dir, names, title, output_name, baseline_setting, xlim)

    # optimizer
    src_dir = "outputs/optimizer/all"
    names = ["baseline+augs+wd+lr+bs", "Adam wd:5e-4+lr:0.01", "Adam wd:5e-4+lr:0.001", "Adam wd:0.01+lr:0.001",
             "SGD wd:5e-4+lr:0.01", "SGD wd:5e-4+lr:0.001", "SGD wd:0.01+lr:0.01", "SGD wd:0.001+lr:0.01"]
    dir_names = ["baseline_norm_crop_flip_rotate5_wd5e-4", "baseline_norm_crop_flip_rotate5_wd5e-4_bs128_adam",
                 "baseline_norm_crop_flip_rotate5_wd5e-4_bs128_adam_lr0.001", "baseline_norm_crop_flip_rotate5_wd0.01_bs128_adam_lr0.001",
                 "baseline_norm_crop_flip_rotate5_wd5e-4_bs128_sgd_lr0.01", "baseline_norm_crop_flip_rotate5_wd5e-4_bs128_sgd_lr0.001",
                 "baseline_norm_crop_flip_rotate5_wd0.01_bs128_sgd_lr0.01", "baseline_norm_crop_flip_rotate5_wd0.001_bs128_sgd_lr0.01"]
    title = "Testset Accuracy"
    output_name = "optimizer_bar.jpg"
    baseline_setting = "AdamW wd:5e-4+lr:0.01"
    xlim = [55, 92]
    extra_data_from_log(src_dir, dir_names, output_dir, names, title, output_name, baseline_setting, xlim)

    # architecture
    src_dir = "outputs/model_arch/all"
    names = ["baseline+augs+wd+lr+bs+opt", "Conv2d_BN(256, 512)\n3.96M Params", "Conv2d_BN+3fc\n4.12M Params",
             "Conv2d_BN+3fc+2drops(0.2)\n4.12M Params", "Conv2d_BN+3fc+2drops(0.4)\n4.12M Params",
             "Conv2d_BN+3fc+3drops(0.2)\n4.12M Params", "Conv2d_BN+3fc+3drops(0.4)\n4.12M Params",
             "Conv2d_BN+1fc+1drop(0.2)\n3.96M Params", "Channels+3fc+3drops(0.2)\nCi=(64, 128, 326)\n4.77M Params"
             ]
    dir_names = ["baseline_norm_crop_flip_rotate5_wd5e-4", "allAugs_AdamW_lr0.01_WeightDecay5e-4_batchSize128_extraConvBn",
                 "allAugs_AdamW_lr0.01_WeightDecay5e-4_batchSize128_extraConvBn_3fcs",
                 "allAugs_AdamW_lr0.01_WeightDecay5e-4_batchSize128_extraConvBn_3fcs_2dropouts",
                 "allAugs_AdamW_lr0.01_WeightDecay5e-4_batchSize128_extraConvBn_3fcs_2dropout0.4",
                 "allAugs_AdamW_lr0.01_WeightDecay5e-4_batchSize128_extraConvBn_3fcs_3dropouts0.2",
                 "allAugs_AdamW_lr0.01_WeightDecay5e-4_batchSize128_extraConvBn_3fcs_3dropout0.4",
                 "allAugs_AdamW_lr0.01_WeightDecay5e-4_batchSize128_extraConvBn_1fcs_1dropout0.2",
                 "allAugs_AdamW_lr0.01_WeightDecay5e-4_batchSize128_extraConvBn_3fcs_3dropouts0.2_changedChannels"
                 ]
    title = "Testset Accuracy"
    output_name = "model_arch_bar.jpg"
    baseline_setting = "2.77M Params"
    xlim = [85, 91]
    extra_data_from_log(src_dir, dir_names, output_dir, names, title, output_name, baseline_setting, xlim)


    # train 200 epochs
    src_dir = "outputs/200epochs/all"
    names = ["baseline+augs+wd+lr+bs+opt", "Conv2d_BN+3fc+3drops(0.2)\nwd:5e-4\n4.12M Params",
             "baseline+augs+wd+lr+bs+opt\nwd:3e-3\n2.77M Params", "Conv2d_BN+3fc+3drops(0.2)\nwd:3e-3\n4.12M Params"]
    dir_names = ["allAugs_AdamW_lr0.01_WeightDecay5e-4_batchSize128_train200ep",
                 "allAugs_AdamW_lr0.01_WeightDecay5e-4_batchSize128_extraConvBn_3fcs_3drouput0.2_train200ep",
                 "allAugs_AdamW_lr0.01_WeightDecay3e-3_batchSize128_train200ep",
                 "allAugs_AdamW_lr0.01_WeightDecay3e-3_batchSize128_extraConvBn_3fcs_3drouput0.2_train200ep"
                 ]
    title = "Testset Accuracy"
    output_name = "train_200epochs_bar.jpg"
    baseline_setting = "wd:5e-4\n2.77M Params"
    xlim = [91, 92.5]
    extra_data_from_log(src_dir, dir_names, output_dir, names, title, output_name, baseline_setting, xlim)