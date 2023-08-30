import os
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import ToTensor
from inference_utils import ImageSequenceReader, ImageSequenceWriter
from concurrent.futures import ThreadPoolExecutor
import tqdm
import torch
from model.model import MattingNetwork
import datetime


class RvmImageSequence:
    def __init__(self, ImageSequence_dir, save_dir="", weight_file="checkpoint/epoch-14.pth", backbone='mobilenetv3',  video_downsample_ratio=1,
                 bgr=None):

        self.backbone = backbone
        self.device = 'cuda'
        self.ImageSequence_dir = ImageSequence_dir
        self.num_workers = 4
        self.weight_file = weight_file

        self.bgr = torch.tensor([0.47, 1., 0.6]).view(3, 1, 1).to(self.device)  # 绿

        if save_dir == "":
            os.makedirs(self.ImageSequence_dir + "_pre", exist_ok=True)
            self.save_dir = self.ImageSequence_dir + "_pre"
        else:
            self.save_dir = save_dir

        self.inference()

    def save_path(self):
        return self.save_dir

    def inference(self):
        model = MattingNetwork(self.backbone).to(self.device)
        model.load_state_dict(torch.load(self.weight_file))

        for dataset in sorted(os.listdir(self.ImageSequence_dir)):
            if os.path.isdir(os.path.join(self.ImageSequence_dir, dataset)):

                save_path = os.path.join(self.save_dir, dataset)
                os.makedirs(save_path, exist_ok=True)

                for clip in sorted(os.listdir(os.path.join(self.ImageSequence_dir, dataset))):

                    local_path = os.path.join(save_path, clip)
                    os.makedirs(local_path, exist_ok=True)

                    pre_path = os.path.join(self.ImageSequence_dir, dataset, clip, "com")
                    save_path_pha = os.path.join(local_path, "pha")
                    os.makedirs(save_path_pha, exist_ok=True)
                    save_path_fgr = os.path.join(local_path, "fgr")
                    os.makedirs(save_path_fgr, exist_ok=True)

                    reader = ImageSequenceReader(pre_path, transform=ToTensor())
                    writer_fgr = ImageSequenceWriter(save_path_fgr)
                    writer_pha = ImageSequenceWriter(save_path_pha)

                    rec = [None] * 4  # 初始循环记忆（Recurrent States）
                    batch_size = 5
                    for batch_size in range(batch_size, 1, -1):
                        if reader.__len__() % batch_size == 0:
                            break

                    with torch.no_grad():
                        for src in tqdm.tqdm(DataLoader(reader, num_workers=self.num_workers, batch_size=batch_size)):
                            # 输入张量，RGB通道，范围为 0～1
                            fgr, pha, *rec = model(src.to(self.device), *rec)  # 将上一帧的记忆给下一帧
                            fgr = fgr*pha

                            writer_fgr.write(fgr)
                            writer_pha.write(pha)

                    writer_fgr.close()
                    writer_pha.close()


class RvmImage:
    def __init__(self, ImageSequence_dir, save_dir="", weight_file="model/epoch-5.pth", backbone="mobilenetv3", downsample_ratio=1.0,
                 bgr=None):
        self.down_ratio = downsample_ratio
        self.backbone = backbone
        self.device = 'cuda'
        self.ImageSequence_dir = ImageSequence_dir
        self.num_workers = 1
        self.weight_file = weight_file


        self.bgr = torch.tensor([0, 0, 0]).view(3, 1, 1).to(self.device)  # 黑背景

        self._init_dir(save_dir)

        self.inference()

    def _init_dir(self, save_dir):
        def get_time():
            date = datetime.datetime.now()
            month_day = str(date.month) + "-" + str(date.day)
            return month_day

        month_day = get_time()
        save_path = save_dir
        os.makedirs(save_path + f"\\{month_day}_RVM", exist_ok=True)
        self.save_dir = save_path + f"\\{month_day}_RVM"

    def inference(self):
        model = MattingNetwork(self.backbone).to(self.device)
        model.load_state_dict(torch.load(self.weight_file))
        save_path_pha = os.path.join(self.save_dir,  "pha")
        os.makedirs(save_path_pha, exist_ok=True)

        save_path_fgr_pha = os.path.join(self.save_dir,  "fgr x pha")
        os.makedirs(save_path_fgr_pha, exist_ok=True)

        save_path_fgr = os.path.join(self.save_dir, "fgr")
        os.makedirs(save_path_fgr, exist_ok=True)

        reader = ImageSequenceReader(os.path.join(self.ImageSequence_dir), transform=ToTensor())

        writer_fgr_pha = ImageSequenceWriter(save_path_fgr_pha)
        writer_pha = ImageSequenceWriter(save_path_pha)
        writer_fgr = ImageSequenceWriter(save_path_fgr)

        with torch.no_grad():
            for src in tqdm.tqdm(DataLoader(reader, num_workers=0, batch_size=1)):
                rec = [None] * 4  # 初始循环记忆（Recurrent States）
                # 输入张量，RGB通道，范围为 0～1
                fgr, pha, *rec = model(src.to(self.device), *rec, downsample_ratio=self.down_ratio)  # 将上一batch的记忆给下一batch
                fgr_pha = fgr*pha

                writer_fgr_pha.write(fgr_pha)
                writer_fgr.write(fgr)
                writer_pha.write(pha)

        writer_fgr_pha.close()
        writer_fgr.close()
        writer_pha.close()



if __name__ == "__main__":
    # dir = r"F:\__test_datasets\videomatte_512x288"
    # RvmImageSequence(dir)
    down_ratio = 1
    weight_file = "checkpoint/epoch-9.pth"
    backbone = "resnet50"
    img_path = r"C:\Users\11958\Desktop\图片1.png"
    save_path = r"bgr"
    RvmImage(img_path, save_path, weight_file, backbone, downsample_ratio=down_ratio)