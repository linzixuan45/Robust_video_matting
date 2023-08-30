from RVM_Images import RvmImageSequence
from evaluate_videomatte_512x288 import Evaluator
import time


if __name__ == "__main__":
    time_start = time.time()
    weight = "model/rvm_resnet50.pth"
    backbone = "resnet50"
    true_low_resolution_dir = r"F:\__test_datasets\videomatte_512x288"
    model = RvmImageSequence(true_low_resolution_dir, weight_file=weight, backbone=backbone)
    pre_save_dir = model.save_path()
    num_workers = 48
    excel_save_dir = "excel_temp"
    Evaluator(pre_save_dir, true_low_resolution_dir, num_workers, excel_save_dir)

    time_end = time.time()
    print(f"process cost time: {int((time_end - time_start)/60)} min {int(time_end - time_start)%60} sec")
