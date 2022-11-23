import matplotlib.pyplot as plt
import os

def plot_data_augs(output_dir):
    files = [
        "outputs/baseline",
        "outputs/baseline_norm",
        "outputs/baseline_norm_crop",
        "outputs/baseline_norm_crop_flip",
        "outputs/baseline_norm_crop_flip_rotate5"
    ]


    for dir_name in files:
        # create output folders
        if not os.path.exists(os.path.join(output_dir, "loss")):
            os.makedirs(os.path.join(output_dir, "loss"))
        if not os.path.exists(os.path.join(output_dir, "acc")):
            os.makedirs(os.path.join(output_dir, "acc"))

        # search for log file
        fns = os.listdir(dir_name)
        log_fn = None
        for fn in fns:
            if "log" in fn:
                log_fn = fn
                break
        
        # read log file lines
        lines = open(os.path.join(dir_name, log_fn)).readlines()

        train_losses = []
        train_acc = []
        dev_losses = []
        dev_acc = []
        test_loss, test_acc = 0, 0
        best_model_epoch = 0
        epoch_number = 0
        for line in lines:
            if "Epoch Average Loss" in line:
                # Sample from log file:   Epoch #0: Epoch Average Loss 1.72230 - Epoch Acc: 35.28222 - Epoch Training Time: 0.51 min(s)
                pieces = line.split("-")
                train_losses.append(float(pieces[0].split("Loss")[-1].strip()))
                train_acc.append(float(pieces[1].split("Acc:")[-1].strip()))
            elif "Eval Devset" in line:
                # Sample from log file:   Eval Devset: Epoch #0: Average Loss 0.81747 - Epoch Acc: 42.90000 - Epoch Testing Time: 0.018 min(s)
                pieces = line.split("-")
                dev_losses.append(float(pieces[0].split("Loss")[-1].strip()))
                dev_acc.append(float(pieces[1].split("Acc:")[-1].strip()))

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
        
        length = len(train_losses)
        plt.figure(figsize=(15, 7))
        plt.title("Train/Dev/Test Loss Plot")
        plt.xlabel("Epoch #")
        plt.ylabel("Avg Epoch Loss")
        plt.plot(range(length), train_losses, label="Train Loss")
        plt.plot(range(length), dev_losses, label="Dev Loss")
        plt.plot([best_model_epoch], [test_loss], "r+", markersize=20, label=f"Test Loss using Best Model: {str(test_loss)}")
        plt.legend()
        plt.grid()

        plt.savefig(os.path.join(output_dir+"loss", dir_name.split("/")[-1] + "_loss.jpg"))

        plt.figure(figsize=(15, 7))
        plt.title("Train/Dev/Test Accuracy Plot")
        plt.xlabel("Epoch #")
        plt.ylabel("Epoch Accuracy")
        plt.plot(range(length), train_acc, label="Train Accuracy")
        plt.plot(range(length), dev_acc, label="Dev Accuracy")
        plt.plot([best_model_epoch], [test_acc], "r+", markersize=20, label=f"Test Accuracy using Best Model: {str(test_acc)}")
        plt.legend()
        plt.grid()

        plt.savefig(os.path.join(output_dir+"acc", dir_name.split("/")[-1] + "_acc.jpg"))

if __name__ == "__main__":
    plot_data_augs(output_dir="output_plots/data_aug/")