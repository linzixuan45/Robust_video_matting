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
    def __init__(self, video_path, save_path, weight_file="checkpoint/epoch-8.pth", backbone='resnet50',frame_rate=30, video_downsample_ratio=None, bgr=None):
        self.video_frame_rate = frame_rate
        self.backbone = backbone
        self.device = 'cuda'
        self.downsample_ratio = video_downsample_ratio  # 下采样比，根据视频调节
        self.weight_file = weight_file
        self._init_dir(save_path)

        if os.path.isdir(video_path):
            video_img_path = list(map(lambda x: os.path.join(video_path, x), os.listdir(video_path)))
            for tmp in video_img_path:
                if os.path.splitext(tmp)[1] != '.mp4':
                    video_img_path.remove(tmp)

            for video in video_img_path:
                self.get_result(video)

        else:
            self.video_name = self.video_basename(video_path)
            self.get_result(video_path)

    def video_basename(self, video_path):
        basename = os.path.basename(video_path)  # 获取原视频的名字
        basename = os.path.splitext(basename)[0]
        return basename

    def _init_dir(self, save_path):
        month_day = get_time()
        os.makedirs(save_path + f"\\{month_day}_RVM_{backbone}", exist_ok=True)
        self.save_dir = save_path + f"\\{month_day}_RVM_{backbone}"

    def get_result(self, video_path):
        self.video_name = self.video_basename(video_path)
        torch.backends.cudnn.benchmark = True
        model = MattingNetwork(variant=self.backbone).eval().to(self.device)  # 或 variant="resnet50"
        model.load_state_dict(torch.load(self.weight_file))
        reader = VideoReader(video_path, transform=ToTensor())
        print("row video frame ratio is ", reader.frame_rate)
        if reader.frame_rate != self.video_frame_rate:
            assert "the output video frame ratio does not match inpute frame ratio"

        if self.downsample_ratio == None:
            h, w = reader.__getitem__(1).shape[-2], reader.__getitem__(1).shape[-1]
            self.downsample_ratio = auto_downsample_ratio(h, w)
            print("down_sample_ratio is : {}", self.downsample_ratio)

        writer_pha = VideoWriter(self.save_dir + f"\\{self.video_name}-pha.mp4", frame_rate=self.video_frame_rate)
        writer_com = VideoWriter(self.save_dir + f"\\{self.video_name}-com.mp4", frame_rate=self.video_frame_rate)
        writer_fgr = VideoWriter(self.save_dir + f"\\{self.video_name}-fgr.mp4", frame_rate=self.video_frame_rate)

        bgr = torch.tensor([1.0, 0, 0]).view(3, 1, 1).to(self.device)  # 绿背景

        rec = [None] * 4  # 初始循环记忆（Recurrent States）
        with torch.no_grad():
            for src in tqdm.tqdm(DataLoader(reader)):  # 输入张量，RGB通道，范围为 0～1

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

    backbone = ['mobilenetv3', "resnet50", "resnet50"]
    weight = ["model/rvm_mobilenetv3.pth", "model/rvm_resnet50.pth", "checkpoint/epoch-8.pth"]
    down_ratio = 1

    video_path = r"C:\Users\11958\Desktop\实验结果\8-25原始视频"
    frame_rate = 25
    save_path = r"C:\Users\11958\Desktop\实验结果"

    basename = video_basename(video_path)

    i= 0
    for backbone, weight in zip(backbone, weight):
        model = RVM_Video(video_path, save_path, backbone=backbone, weight_file=weight, frame_rate=frame_rate, video_downsample_ratio=down_ratio)
        save_path = save_path + f"_{i}"
        i += 1

    # model = RVM_Video(video_path, save_path, backbone=backbone[1], weight_file=weight[1], frame_rate=frame_rate,
    #                   video_downsample_ratio=down_ratio)

