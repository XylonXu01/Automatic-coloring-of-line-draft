import cv2
from PySide2 import QtCore
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QFileDialog
from PySide2.QtWidgets import QApplication
from PySide2.QtGui import *
import torch
from Generate_Imag import load_image_test
import PySide2.QtXml
from matplotlib import pyplot as plt
from keras.models import load_model
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 加上才能运行
dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
# 加上才能显示图片
QtCore.QCoreApplication.addLibraryPath(os.path.join(os.path.dirname(QtCore.__file__), "plugins"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('./AnimeColorization_Modelv1.h5')
IMG_WIDTH = 256
IMG_HEIGHT = 256


# 前端界面
class Stats:
    num = 0
    picture = ""  # 图片路径
    lujing = ""  # 路径
    page = 0

    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load('test(3).ui')
        self.ui.button.clicked.connect(self.openfile)
        self.ui.button_2.clicked.connect(self.Recognition)
        self.ui.up.clicked.connect(self.uppage)
        self.ui.down.clicked.connect(self.downpage)


    def openfile(self):  # 选择图片并显示
        FileDialog = QFileDialog(self.ui)
        FileDirectory = FileDialog.getOpenFileName(self.ui, "标题")  # 选择目录，返回选中的路径
        self.picture = os.path.abspath(FileDirectory[0])  # 获取图片的路径！！！！！！！！
        self.lujing = os.path.dirname(self.picture)
        print(self.picture)
        print(self.lujing)
        picture = QPixmap(self.picture)
        pictures = picture.copy(512, 0, 512, 512)
        print(type(picture))
        print(pictures)
        self.ui.label.setPixmap(pictures)

    def uppage(self):  # 上页功能
        # 计算当前文件所在位置
        self.page = 0
        num = 0
        for filename in os.listdir(self.lujing):
            if os.path.join(self.lujing, filename) == self.picture:
                break
            self.page += 1
        for filename in os.listdir(self.lujing):
            num += 1
            if num == self.page:
                self.picture = os.path.join(self.lujing, filename)
                break
        print(self.picture)
        mini_picture = QPixmap(self.picture)
        mini_picture = mini_picture.copy(512, 0, 512, 512)
        self.ui.label.setPixmap(mini_picture)

    def downpage(self):  # 下页功能
        # 计算当前文件所在位置
        self.page = 0
        num = -1
        for filename in os.listdir(self.lujing):
            if os.path.join(self.lujing, filename) == self.picture:
                break
            self.page += 1
        for filename in os.listdir(self.lujing):
            if num == self.page:
                self.picture = os.path.join(self.lujing, filename)
                break
            num += 1
        print(self.picture)
        picture = QPixmap(self.picture)
        picture = picture.copy(512, 0, 512, 512)
        self.ui.label.setPixmap(picture)

    def Recognition(self):  # 上色
        input_image, real_image = load_image_test(self.picture)
        if len(input_image.shape) < 4:
            input_image = np.expand_dims(input_image, axis=0)
        if len(real_image.shape) < 4:
            real_image = np.expand_dims(real_image, axis=0)
        prediction = model(input_image, training=True)
        """
        from keras.preprocessing import image
        prediction = image.array_to_img(prediction[0] * 255, scale=False)
        prediction.save("Generate_image.png")
        Generate_picture = QPixmap("Generate_image.png")
        self.ui.label_2.setPixmap(Generate_picture)
        """
        plt.figure(figsize=(5, 5))
        plt.imshow(prediction[0] * 0.5 + 0.5)
        plt.axis('off')
        plt.savefig("Generate_image.png")
        # plt.show()
        Generate_picture = QPixmap("Generate_image.png")
        self.ui.label_2.setPixmap(Generate_picture)


app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec_()
