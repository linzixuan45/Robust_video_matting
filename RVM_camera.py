
import sys
import os.path
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
# from inference_utils import VideoReader, VideoWriter
from model import MattingNetwork
import datetime
import tqdm
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)


class RVM_camera:
    def __init__(self, save_path, weight_file="checkpoint/epoch-8.pth", backbone='resnet50',frame_rate=30, video_downsample_ratio=None, bgr=None):
        self.video_frame_rate = frame_rate
        self.backbone = backbone
        self.device = 'cuda'
        self.downsample_ratio = video_downsample_ratio  # 下采样比，根据视频调节
        self.weight_file = weight_file
        self._init_dir(save_path)

    def _init_dir(self, save_path):
        month_day = get_time()
        os.makedirs(save_path + f"\\{month_day}_RVM_{backbone}", exist_ok=True)
        self.save_dir = save_path + f"\\{month_day}_RVM_{backbone}"

    def camera_matting(self):
        self.video_name = time.time()
        torch.backends.cudnn.benchmark = True
        model = MattingNetwork(variant=self.backbone).eval().to(self.device)  # 或 variant="resnet50"
        model.load_state_dict(torch.load(self.weight_file))

        # writer_pha = VideoWriter(self.save_dir + f"\\{self.video_name}-pha.mp4", frame_rate=self.video_frame_rate)
        # writer_com = VideoWriter(self.save_dir + f"\\{self.video_name}-com.mp4", frame_rate=self.video_frame_rate)
        # writer_fgr = VideoWriter(self.save_dir + f"\\{self.video_name}-fgr.mp4", frame_rate=self.video_frame_rate)

        bgr = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1).to(self.device)  # 绿背景
        rec = [None] * 4  # 初始循环记忆（Recurrent States）
        """------------摄像头设置---------------"""
        cap = cv2.VideoCapture(0)  # 打开默认摄像头采集图像

        width = 1280  # 定义摄像头获取图像宽度
        height = 720  # 定义摄像头获取图像长度

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置宽度
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 设置长度

        while (cap.isOpened()):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ToTensor()(frame)
            frame = frame.unsqueeze(0)
            with torch.no_grad():
                fgr, pha, *rec = model(frame.to(self.device), *rec, self.downsample_ratio)  # 将上一帧的记忆给下一帧
                com = fgr * pha + (1 - pha) * bgr
            com = com.squeeze(0)
            com = com.cpu().numpy()
            com_img = com*255
            com_img = np.array(com_img,dtype=np.uint8)# 1,3,480,640
            com_img = com_img.transpose((1,2,0))
            com_img = cv2.cvtColor(com_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("frame", com_img)

            c = cv2.waitKey(int(1/self.video_frame_rate * 1000))  # 单位ms  每一帧25ms 也就是 一秒40fps
            if c == 27:
                sys.exit()

            else:
                pass
        cap.release()


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

    frame_rate = 1000
    save_path = r"C:\Users\11958\Desktop\实验结果"

    i= 0

    model = RVM_camera( save_path, backbone=backbone[2], weight_file=weight[2], frame_rate=frame_rate, video_downsample_ratio=down_ratio)
    model.camera_matting()

    # model = RVM_Video(video_path, save_path, backbone=backbone[1], weight_file=weight[1], frame_rate=frame_rate,
    #                   video_downsample_ratio=down_ratio)

