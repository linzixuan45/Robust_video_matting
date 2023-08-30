import os.path
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from inference_utils import VideoReader, VideoWriter
from model import MattingNetwork
import datetime
import tqdm


class RVM_Video:
    def __init__(self, fgr_path, alpha_path, bgr_path, save_path ):
        self.fgr_path = fgr_path
        self.bgr_path = bgr_path
        self.alpha_path = alpha_path
        self.device = 'cuda'
        self._init_dir(save_path)

        self.get_result()

    def video_basename(self, video_path):
        basename = os.path.basename(video_path)  # 获取原视频的名字
        basename = os.path.splitext(basename)[0]
        return basename

    def _init_dir(self, save_path):
        month_day = get_time()
        os.makedirs(save_path + f"\\{month_day}_RVM", exist_ok=True)
        self.save_dir = save_path + f"\\{month_day}_RVM"

    def get_result(self):
        self.video_name = self.video_basename(self.fgr_path)
        fgr_reader = VideoReader(self.fgr_path, transform=ToTensor())
        pha_reader = VideoReader(self.alpha_path, transform=ToTensor())
        bgr_reader = VideoReader(self.bgr_path, transform=ToTensor())
        writer_save = VideoWriter(self.save_dir + f"\\{self.video_name}-pha.mp4", frame_rate=60)

        with torch.no_grad():
            for src in tqdm.tqdm(DataLoader(fgr_reader), DataLoader(pha_reader)):  # 输入张量，RGB通道，范围为 0～1

                fgr, pha, *rec = model(src.to(self.device), *rec, self.downsample_ratio)  # 将上一帧的记忆给下一帧
                com = fgr * pha + (1 - pha) * bgr

                writer_pha.write(pha)
                writer_com.write(com)
                writer_fgr.write(fgr)

        writer_pha.close()
        writer_com.close()
        writer_fgr.close()

    def save_path(self):
        return self.save_dir + f"\\{self.video_name}-fgr.mp4"


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


def get_time():
    date = datetime.datetime.now()
    month_day = str(date.month) + "-" + str(date.day)
    return month_day

def video_basename(video_path):
    basename = os.path.basename(video_path)  # 获取原视频的名字
    basename = os.path.splitext(basename)[0]
    return basename

if __name__ == "__main__":
    # video_path = r"C:\Users\11958\Desktop\实验结果\原始视频\2.mp4"
    # save_path = r"C:\Users\11958\Desktop\实验结果"

    fgr_path = ""




    # for backbone, weight in zip(backbone, weight):
    #     model = RVM_Video(video_path, save_path, backbone=backbone, weight_file=weight, video_downsample_ratio=down_ratio)
    #     video_path = model.save_path()
    #     save_path = save_path + f"\\{get_time()}_RVM"

