import sys
import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPainter,QColor


from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
    QFileDialog, QMessageBox, QProgressBar, QVBoxLayout, QHBoxLayout,
    QRadioButton, QButtonGroup, QGroupBox, QSplashScreen, QMenuBar, QAction
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont


# 获取应用程序的路径
if getattr(sys, 'frozen', False):
    # 打包后的应用程序
    application_path = os.path.dirname(sys.executable)
else:
    # 开发环境
    application_path = os.path.dirname(os.path.abspath(__file__))


# 设置日志记录
log_file_path = os.path.join(application_path, 'drbp_edp.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.ERROR,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 tokenizer，指定模型路径为 model 文件夹中的预训练模型
model_checkpoint = os.path.join(application_path, "model", "esm2_t33_650M_UR50D")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
except Exception as e:
    logging.error("Error loading tokenizer: %s", e)
    QMessageBox.critical(None, "Tokenizer Error", f"Failed to load tokenizer: {e}")
    sys.exit(1)

# 定义数据集类
class ProteinSequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=1000):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoded_sequence = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = encoded_sequence['input_ids'].squeeze(0)
        attention_mask = encoded_sequence['attention_mask'].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sequence_length": len(sequence)
        }

# 定义模型
class MultiPathProteinClassifier(nn.Module):
    def __init__(self, model_checkpoint):
        super(MultiPathProteinClassifier, self).__init__()
        try:
            self.esm2 = AutoModel.from_pretrained(model_checkpoint, ignore_mismatched_sizes=True)
        except Exception as e:
            logging.error("Error loading ESM2 model: %s", e)
            raise e

        # 路径1: Transformer + CNN
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1280, nhead=8, batch_first=True, dropout=0.6),
            num_layers=1
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1280, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool1d(output_size=500),
            nn.Dropout(0.2),

            nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool1d(output_size=200),
            nn.Dropout(0.2),

            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool1d(output_size=20),
            nn.Dropout(0.2),
        )

        # 路径2: BiLSTM + Attention
        self.bilstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        self.attention_pool = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.6)
        )

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(896, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.6),
        )

        # 第一阶段和第二阶段的分类器
        self.classifier_stage1 = nn.Linear(256, 1)  # 核酸结合与非核酸结合
        self.classifier_stage2 = nn.Linear(256, 1)  # DNA结合与RNA结合

        # 当前训练阶段
        self.current_stage = 1

    def forward(self, input_ids, attention_mask):
        # ESM-2输出
        shared_output = self.esm2(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 路径1: Transformer + CNN
        x1 = self.transformer(shared_output)
        x1 = x1.permute(0, 2, 1)
        x1 = self.cnn(x1).view(x1.size(0), -1)

        # 路径2: BiLSTM + Attention
        x2, _ = self.bilstm(shared_output)
        attention_output, _ = self.attention(x2, x2, x2)
        x2 = attention_output.mean(dim=1)
        x2 = self.attention_pool(x2)

        # 特征融合
        features = torch.cat([x1, x2], dim=1)
        features = self.feature_fusion(features)

        # 根据当前阶段使用相应的分类器
        if self.current_stage == 1:
            return self.classifier_stage1(features)
        else:
            return self.classifier_stage2(features)

    def set_stage(self, stage):
        self.current_stage = stage

# 定义加载模型的线程
class LoadModelThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool)  # 传递是否成功加载

    def __init__(self, model_checkpoint, parent=None):
        super(LoadModelThread, self).__init__(parent)
        self.model_checkpoint = model_checkpoint
        self.stage1_model = None
        self.stage2_model = None
        self.success = False

    def run(self):
        try:
            # 加载阶段1模型
            self.stage1_model = MultiPathProteinClassifier(self.model_checkpoint)
            state_dict_stage1 = torch.load(os.path.join(application_path, 'model', 'best_model_stage1.pth'), map_location=device)
            # 处理 state_dict 中的 'module.' 前缀
            new_state_dict_stage1 = {}
            for k, v in state_dict_stage1.items():
                if k.startswith('module.'):
                    k = k[7:]  # 去掉 'module.' 前缀
                new_state_dict_stage1[k] = v
            self.stage1_model.load_state_dict(new_state_dict_stage1)
            self.stage1_model.to(device)
            self.stage1_model.eval()
            self.stage1_model.set_stage(1)
            self.progress.emit(50)

            # 加载阶段2模型
            self.stage2_model = MultiPathProteinClassifier(self.model_checkpoint)
            state_dict_stage2 = torch.load(os.path.join(application_path, 'model', 'best_model_stage2.pth'), map_location=device)
            # 处理 state_dict 中的 'module.' 前缀
            new_state_dict_stage2 = {}
            for k, v in state_dict_stage2.items():
                if k.startswith('module.'):
                    k = k[7:]  # 去掉 'module.' 前缀
                new_state_dict_stage2[k] = v
            self.stage2_model.load_state_dict(new_state_dict_stage2)
            self.stage2_model.to(device)
            self.stage2_model.eval()
            self.stage2_model.set_stage(2)
            self.progress.emit(100)
            self.success = True
            self.finished.emit(self.success)
        except Exception as e:
            logging.error("Error loading models: %s", e)
            self.success = False
            self.finished.emit(self.success)

# 定义预测的线程
class PredictionThread(QThread):
    progress = pyqtSignal(int)  # 进度百分比
    result = pyqtSignal(list)  # 最终结果列表
    error = pyqtSignal(str)  # 错误信息

    def __init__(self, task, sequences, tokenizer, stage1_model, stage2_model, parent=None):
        super(PredictionThread, self).__init__(parent)
        self.task = task  # 1或2
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.model = stage1_model if task == 1 else stage2_model  # 根据任务选择模型

    def run(self):
        try:
            dataset = ProteinSequenceDataset(self.sequences, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            results = []

            with torch.no_grad():
                total_sequences = len(self.sequences)
                for idx, batch in enumerate(dataloader):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    sequence_length = batch['sequence_length'].item()
                    sequence = self.sequences[idx]

                    # 统一使用选定的模型进行预测
                    logits = self.model(input_ids, attention_mask)
                    prob = torch.sigmoid(logits).item()
                    pred = int(prob >= 0.5)

                    # 根据任务类型生成预测结果
                    if self.task == 1:
                        if pred == 1:
                            prediction = 'Nucleic acid-binding protein'
                            final_prob = prob
                        else:
                            prediction = 'Non-nucleic acid-binding protein'
                            final_prob = 1 - prob
                    else:
                        if pred == 0:
                            prediction = 'DNA-binding protein'
                            final_prob = 1 - prob
                        else:
                            prediction = 'RNA-binding protein'
                            final_prob = prob

                    # 确保概率在 [0, 1] 之间
                    final_prob = max(0.0, min(final_prob, 1.0))

                    # 添加序列长度和警告信息
                    warnings = []
                    if sequence_length < 50:
                        warnings.append('Length less than 50; prediction may be less reliable.')
                    if sequence_length > 1000:
                        warnings.append('Length greater than 1000; prediction may be less reliable.')
                    length_warning = ' '.join(warnings)

                    results.append({
                        'Sequence': sequence,
                        'Sequence Length': sequence_length,
                        'Prediction': prediction,
                        'Probability': final_prob,
                        'Warning': length_warning
                    })

                    # 更新进度
                    progress = int((idx + 1) / total_sequences * 100)
                    self.progress.emit(progress)

            self.result.emit(results)
        except Exception as e:
            logging.error("Error during prediction: %s", e)
            self.error.emit(str(e))

class CustomSplashScreen(QSplashScreen):
    def __init__(self, width=800, height=600, parent=None):
        # 创建一个指定尺寸的QPixmap，并填充为薰衣草紫色
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("#7F77CB"))  # 薰衣草紫色

        super(CustomSplashScreen, self).__init__(pixmap, Qt.WindowStaysOnTopHint)

        # 设置字体和颜色
        self.title_font = QFont("Arial", 60, QFont.Bold)  # 第一排字体
        self.message_font = QFont("Arial", 28)  # 第二排字体
        self.text_color = QColor("white")  # 白色文本

        # 设置窗口大小
        self.setFixedSize(width, height)

    def paintEvent(self, event):
        super(CustomSplashScreen, self).paintEvent(event)
        painter = QPainter(self)

        # 设置字体和颜色
        painter.setFont(self.title_font)
        painter.setPen(self.text_color)

        # 绘制第一排文本（DRBP_EDP）
        title_text = "DRBP_EDP"
        painter.drawText(self.rect(), Qt.AlignTop | Qt.AlignHCenter, title_text)

        # 设置第二排字体
        painter.setFont(self.message_font)

        # 绘制第二排文本（Loading models, please wait...）
        message_text = "Loading models, please wait..."
        painter.drawText(self.rect(), Qt.AlignBottom | Qt.AlignHCenter, message_text)

        painter.end()

    # 重写以下事件处理方法，忽略所有用户输入，防止点击关闭启动窗口
    def mousePressEvent(self, event):
        pass  # 忽略鼠标点击事件

    def mouseReleaseEvent(self, event):
        pass  # 忽略鼠标释放事件

    def mouseDoubleClickEvent(self, event):
        pass  # 忽略鼠标双击事件

    def keyPressEvent(self, event):
        pass  # 忽略键盘按下事件

    def keyReleaseEvent(self, event):
        pass  # 忽略键盘释放事件


# 定义主窗口
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DRBP-EDP Predictor")

        self.setGeometry(100, 100, 900, 700)
        self.setMinimumSize(800, 600)

        self.stage1_model = None
        self.stage2_model = None

        self.initUI()

    def center(self):
        """
        将窗口居中显示在屏幕上。
        """
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        window_geometry = self.frameGeometry()
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())

    def initUI(self):
        # 设置字体
        self.font_label_large = QFont("Arial", 14)  # 增大字体
        self.font_label_normal = QFont("Arial", 14)  # 普通字体
        self.font_button_large = QFont("Arial", 12)
        self.font_button_normal = QFont("Arial", 12)
        self.font_text = QFont("Arial", 12)

        # 创建主布局
        main_layout = QVBoxLayout()

        # 添加自定义标题标签
        self.title_label = QLabel("DRBP-EDP Predictor", self)
        self.title_label.setFont(QFont("Arial", 15, QFont.Bold))  # 设置字体大小为15并加粗
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label, stretch=0)  # 将标题添加到布局顶部

        # 创建菜单栏
        self.menu_bar = QMenuBar(self)
        help_action = QAction("Help", self)
        help_action.triggered.connect(self.showHelp)
        self.menu_bar.addAction(help_action)  # 直接添加Help，不使用子菜单

        # 设置菜单栏字体大小
        self.menu_bar.setFont(QFont("Arial", 14, QFont.Bold))  # 设置菜单栏整体字体大小和加粗
        help_action.setFont(QFont("Arial", 14, QFont.Bold))  # 设置“Help”字体大小


        # 输入方式选择
        self.input_method_group = QButtonGroup(self)

        self.rb_single = QRadioButton("Single Sequence")
        self.rb_file = QRadioButton("File Input")
        self.rb_single.setChecked(True)

        self.input_method_group.addButton(self.rb_single)
        self.input_method_group.addButton(self.rb_file)

        input_method_layout = QHBoxLayout()
        input_method_layout.addWidget(self.rb_single)
        input_method_layout.addWidget(self.rb_file)

        input_method_box = QGroupBox("Select Input Method")
        input_method_box.setLayout(input_method_layout)
        input_method_box.setFont(QFont("Arial", 14))  # 增大字体
        input_method_box.setFixedHeight(100)  # 减少组框高度到80

        # 设置单选按钮的字体大小
        self.rb_single.setFont(QFont("Arial", 12))
        self.rb_file.setFont(QFont("Arial", 12))


        # 添加任务选择组件
        self.task_group = QButtonGroup(self)
        self.rb_task1 = QRadioButton(" NABPs  /  Non-NABPs")
        self.rb_task2 = QRadioButton(" DBPs  /  RBPs")
        self.rb_task1.setChecked(True)

        self.task_group.addButton(self.rb_task1)
        self.task_group.addButton(self.rb_task2)

        task_layout = QHBoxLayout()
        task_layout.addWidget(self.rb_task1)
        task_layout.addWidget(self.rb_task2)

        task_box = QGroupBox("Select Classification Task")
        task_box.setLayout(task_layout)
        task_box.setFont(QFont("Arial", 14))
        task_box.setFixedHeight(100)



        # 单序列输入
        self.label_sequence = QLabel("Enter Protein Sequence:")
        self.label_sequence.setFont(self.font_label_normal)
        self.text_sequence = QTextEdit()
        self.text_sequence.setFont(self.font_text)
        self.text_sequence.setFixedHeight(100)

        # 文件输入
        self.label_file = QLabel("Select Input File:")
        self.label_file.setFont(self.font_label_normal)

        self.entry_file = QLineEdit()
        self.entry_file.setFont(self.font_text)
        self.entry_file.setFixedHeight(80)  # 增加输入框高度

        self.btn_browse = QPushButton("Browse")
        self.btn_browse.setFont(self.font_button_normal)
        self.btn_browse.setFixedHeight(80)  # 增加按钮高度
        self.btn_browse.clicked.connect(self.selectFile)

        # 输出文件选择
        self.label_output = QLabel("Output File Path:")
        self.label_output.setFont(self.font_label_normal)

        self.entry_output = QLineEdit()
        self.entry_output.setFont(self.font_text)
        self.entry_output.setFixedHeight(80)  # 增加输入框高度
        self.entry_output.setText(os.path.join(application_path, 'output.csv'))

        self.btn_output = QPushButton("Select")
        self.btn_output.setFont(self.font_button_normal)
        self.btn_output.setFixedHeight(80)  # 增加按钮高度
        self.btn_output.clicked.connect(self.selectOutputFile)


        # 预测和清除按钮
        self.btn_predict = QPushButton("Predict")
        self.btn_predict.setFont(self.font_button_normal)
        self.btn_predict.clicked.connect(self.runPrediction)

        self.btn_clear_input = QPushButton("Clear Input")
        self.btn_clear_input.setFont(self.font_button_normal)
        self.btn_clear_input.clicked.connect(self.clearInput)

        self.btn_clear_output = QPushButton("Clear Output")
        self.btn_clear_output.setFont(self.font_button_normal)
        self.btn_clear_output.clicked.connect(self.clearOutput)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.btn_predict)
        buttons_layout.addWidget(self.btn_clear_input)
        buttons_layout.addWidget(self.btn_clear_output)

        # 状态标签和进度条
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setFont(self.font_label_normal)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)

        status_layout = QHBoxLayout()
        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.progress_bar)

        # 单序列结果显示
        self.label_result = QLabel("Prediction Result:")
        self.label_result.setFont(self.font_label_normal)
        self.text_output = QTextEdit()
        self.text_output.setFont(self.font_text)
        self.text_output.setReadOnly(True)

        self.btn_copy = QPushButton("Copy Result")
        self.btn_copy.setFont(self.font_button_normal)
        self.btn_copy.clicked.connect(self.copyResult)

        result_layout = QHBoxLayout()
        result_layout.addWidget(self.text_output)
        result_layout.addWidget(self.btn_copy)

        # 布局组合
        main_layout = QVBoxLayout()
        main_layout.setMenuBar(self.menu_bar)  # 将菜单栏添加到主布局
        main_layout.addWidget(input_method_box)
        # 将任务选择框添加到主布局
        main_layout.addWidget(task_box)

        # Stack for single or file input
        self.single_input_layout = QVBoxLayout()
        self.single_input_layout.addWidget(self.label_sequence)
        self.single_input_layout.addWidget(self.text_sequence)

        self.file_input_layout = QVBoxLayout()
        self.file_input_layout.addWidget(self.label_file)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.entry_file)
        file_layout.addWidget(self.btn_browse)
        self.file_input_layout.addLayout(file_layout)
        self.file_input_layout.addWidget(self.label_output)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.entry_output)
        output_layout.addWidget(self.btn_output)
        self.file_input_layout.addLayout(output_layout)

        self.input_stack = QVBoxLayout()
        self.input_stack.addLayout(self.single_input_layout)
        self.input_stack.addLayout(self.file_input_layout)

        main_layout.addLayout(self.input_stack)
        main_layout.addLayout(buttons_layout)
        main_layout.addLayout(status_layout)
        main_layout.addWidget(self.label_result)
        main_layout.addLayout(result_layout)

        self.setLayout(main_layout)

        # 连接单选按钮以切换输入方式
        self.input_method_group.buttonClicked.connect(self.toggleInputMethod)
        self.toggleInputMethod()

    def toggleInputMethod(self):
        if self.rb_single.isChecked():
            self.label_sequence.show()
            self.text_sequence.show()
            self.label_file.hide()
            self.entry_file.hide()
            self.btn_browse.hide()
            self.label_output.hide()
            self.entry_output.hide()
            self.btn_output.hide()

            # 显示预测结果部分
            self.label_result.show()
            self.text_output.show()
            self.btn_copy.show()

            # 调整字体大小
            self.rb_single.setFont(self.font_label_large)
            self.rb_file.setFont(self.font_label_normal)

            # 显示“Clear Input”和“Clear Output”按钮
            self.btn_clear_input.show()
            self.btn_clear_output.show()
        else:
            self.label_sequence.hide()
            self.text_sequence.hide()
            self.label_file.show()
            self.entry_file.show()
            self.btn_browse.show()
            self.label_output.show()
            self.entry_output.show()
            self.btn_output.show()

            # 隐藏预测结果部分
            self.label_result.hide()
            self.text_output.hide()
            self.btn_copy.hide()

            # 调整字体大小
            self.rb_file.setFont(self.font_label_large)
            self.rb_single.setFont(self.font_label_normal)

            # 隐藏“Clear Input”和“Clear Output”按钮
            self.btn_clear_input.hide()
            self.btn_clear_output.hide()

    def selectFile(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input File",
            "",
            "FASTA Files (*.fasta *.fa);;TSV Files (*.tsv);;TXT Files (*.txt)",
            options=options
        )
        if file_path:
            self.entry_file.setText(file_path)

    def selectOutputFile(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output File",
            "",
            "CSV Files (*.csv)",
            options=options
        )
        if file_path:
            self.entry_output.setText(file_path)

    def clearInput(self):
        if self.rb_single.isChecked():
            self.text_sequence.clear()

    def clearOutput(self):
        self.text_output.clear()

    def copyResult(self):
        result_text = self.text_output.toPlainText()
        if result_text.strip():
            clipboard = QApplication.clipboard()
            clipboard.setText(result_text)
            QMessageBox.information(self, "Copy to Clipboard", "Result has been copied to clipboard.")

    def showErrorMessage(self, message):
        QMessageBox.critical(self, "Error", message, QMessageBox.Ok)
    def showInfoMessage(self, message):
        QMessageBox.information(self, "Information", message, QMessageBox.Ok)

    def loadModels(self, splash, main_window):
        # 启动模型加载线程
        model_checkpoint = os.path.join(application_path, "model", "esm2_t33_650M_UR50D")

        self.load_thread = LoadModelThread(model_checkpoint)
        self.load_thread.progress.connect(self.updateLoadProgress)
        self.load_thread.finished.connect(lambda success: self.onModelsLoaded(success, splash, main_window))
        self.load_thread.start()

    def updateLoadProgress(self, value):
        # 可以在启动窗口上显示进度条或文字
        pass  # 由于启动窗口已经显示加载信息，此处暂不处理

    def onModelsLoaded(self, success, splash, main_window):
        if success:
            self.stage1_model = self.load_thread.stage1_model
            self.stage2_model = self.load_thread.stage2_model

            # 关闭启动窗口
            splash.finish(main_window)
            splash.close()

            # 显示信息提示框，居中于屏幕
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Information")
            msg_box.setText("Models loaded successfully.")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setWindowModality(Qt.ApplicationModal)
            msg_box.setDefaultButton(QMessageBox.Ok)
            msg_box.setAttribute(Qt.WA_DeleteOnClose)

            # 确保消息框在屏幕中央
            msg_box.setGeometry(
                QApplication.desktop().screen().rect().center().x() - 200,
                QApplication.desktop().screen().rect().center().y() - 100,
                400,
                200
            )

            # 显示消息框并等待用户点击“OK”
            if msg_box.exec_() == QMessageBox.Ok:
                # 居中并显示主窗口
                self.center()
                main_window.show()
        else:
            self.showErrorMessage("Failed to load models. Please check the log file for details.")
            sys.exit(1)

    def validateSequence(self, seq):
        # 移除所有空格和回车符
        valid_aa = 'ACDEFGHIKLMNPQRSTVWY'
        cleaned_seq = ''.join(seq.upper().split())  # 移除所有空格和回车符
        if not cleaned_seq:
            return False, ""
        if all(c in valid_aa for c in cleaned_seq):
            return True, cleaned_seq
        else:
            # 找出不合法的字符
            invalid_chars = set(c for c in cleaned_seq if c not in valid_aa)
            return False, ''.join(invalid_chars)

    def parseFASTA(self, file_path):
        sequences = []
        try:
            with open(file_path, 'r') as f:
                sequence = ''
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if sequence:
                            sequences.append(sequence)
                            sequence = ''
                    else:
                        sequence += line
                if sequence:
                    sequences.append(sequence)
            return sequences
        except Exception as e:
            logging.error("Error parsing FASTA file: %s", e)
            self.showErrorMessage(f"Failed to parse FASTA file: {e}")
            return []

    def parseTXT(self, file_path):
        sequences = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sequences.append(line)
            return sequences
        except Exception as e:
            logging.error("Error parsing TXT file: %s", e)
            self.showErrorMessage(f"Failed to parse TXT file: {e}")
            return []

    def parseTSV(self, file_path):
        try:
            df = pd.read_csv(file_path, sep='\t')
            if 'Sequence' not in df.columns:
                self.showErrorMessage("TSV file must contain a 'Sequence' column.")
                return []
            sequences = df['Sequence'].dropna().astype(str).tolist()
            return sequences
        except Exception as e:
            logging.error("Error parsing TSV file: %s", e)
            self.showErrorMessage(f"Failed to parse TSV file: {e}")
            return []

    def runPrediction(self):
        # 获取当前任务选择
        current_task = 1 if self.rb_task1.isChecked() else 2

        if not self.stage1_model or not self.stage2_model:
            self.showErrorMessage("Models are not loaded yet.")
            return

        if self.rb_single.isChecked():
            input_text = self.text_sequence.toPlainText().strip()
            if not input_text:
                self.showErrorMessage("Please enter a protein sequence.")
                return
            is_valid, processed_seq_or_invalid = self.validateSequence(input_text)
            if not is_valid:
                self.showErrorMessage(
                    f"Invalid sequence characters detected: {processed_seq_or_invalid}\n"
                    "Please ensure all sequences only contain valid amino acid characters (ACDEFGHIKLMNPQRSTVWY)."
                )
                return
            sequences = [processed_seq_or_invalid]  # 使用清理后的序列
            single_sequence = True
        else:
            file_path = self.entry_file.text().strip()
            if not file_path:
                self.showErrorMessage("Please select an input file.")
                return
            if not os.path.isfile(file_path):
                self.showErrorMessage("The selected file does not exist.")
                return

            if file_path.endswith('.fasta') or file_path.endswith('.fa'):
                sequences = self.parseFASTA(file_path)
            elif file_path.endswith('.tsv'):
                sequences = self.parseTSV(file_path)
            elif file_path.endswith('.txt'):
                sequences = self.parseTXT(file_path)
            else:
                self.showErrorMessage(
                    "Unsupported file format. Only FASTA (.fasta, .fa), TSV (.tsv), and TXT (.txt) files are supported.")
                return

            if not sequences:
                self.showErrorMessage("No valid sequences found in the selected file.")
                return

            # 验证所有序列，清理并跳过无效字符
            cleaned_sequences = []
            invalid_sequences = []
            for seq in sequences:
                is_valid, processed_seq_or_invalid = self.validateSequence(seq)
                if is_valid:
                    cleaned_sequences.append(processed_seq_or_invalid)
                else:
                    invalid_sequences.append(seq)

            if invalid_sequences:
                # 为避免界面过于拥挤，仅显示前10个无效序列
                displayed_invalid = ', '.join(invalid_sequences[:10])
                if len(invalid_sequences) > 10:
                    displayed_invalid += '...'
                self.showErrorMessage(
                    f"Invalid sequences detected (ignoring invalid characters):\n{displayed_invalid}\n"
                    "Spaces and newlines are ignored. Please ensure all sequences only contain valid amino acid characters (ACDEFGHIKLMNPQRSTVWY)."
                )
                # 继续处理有效的序列
                if not cleaned_sequences:
                    self.showErrorMessage("No valid sequences to predict after removing invalid characters.")
                    return

            # 检查序列数量
            max_sequences = 1000
            if len(cleaned_sequences) > max_sequences:
                self.showErrorMessage(
                    f"The input file contains {len(cleaned_sequences)} valid sequences, which exceeds the maximum allowed ({max_sequences}). "
                    f"Please select a file with up to {max_sequences} sequences."
                )
                return

            output_path = self.entry_output.text().strip()
            if not output_path:
                output_path = os.path.join(application_path, 'output.csv')

            sequences = cleaned_sequences
            single_sequence = False

        # 启动预测线程
        self.btn_predict.setEnabled(False)
        self.lbl_status.setText("Predicting, please wait...")
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2196F3; }")  # 蓝色


        self.prediction_thread = PredictionThread(
            task=current_task,
            sequences=sequences,
            tokenizer=tokenizer,
            stage1_model=self.stage1_model,
            stage2_model=self.stage2_model
        )
        self.prediction_thread.progress.connect(self.updateProgress)
        if single_sequence:
            self.prediction_thread.result.connect(self.displaySingleResult)
        else:
            self.prediction_thread.result.connect(
                lambda res: self.saveBatchResults(res, self.entry_output.text().strip()))
        self.prediction_thread.error.connect(self.showErrorMessage)
        self.prediction_thread.finished.connect(self.onPredictionFinished)
        self.prediction_thread.start()

    def updateProgress(self, value):
        self.progress_bar.setValue(value)
        if value < 100:
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2196F3; }")  # 蓝色
        else:
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")  # 绿色

    def displaySingleResult(self, results):
        if results:
            result = results[0]
            output = (
                f"Sequence Length: {result['Sequence Length']}\n"
                f"Prediction: {result['Prediction']}\n"
                f"Probability: {result['Probability']:.4f}\n"
                f"Warning: {result['Warning']}"
            )
            self.text_output.setText(output)
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")  # 绿色
        else:
            self.text_output.setText("No results to display.")

    def saveBatchResults(self, results, output_path):
        if not output_path.endswith('.csv'):
            output_path += '.csv'
        try:
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_path, index=False)
            self.showInfoMessage(f"Predictions saved to {output_path}")
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")  # 绿色
        except Exception as e:
            logging.error("Error saving results: %s", e)
            self.showErrorMessage(f"Failed to save results: {e}")

    def onPredictionFinished(self):
        self.btn_predict.setEnabled(True)
        self.lbl_status.setText("Ready")
        self.progress_bar.setValue(100)

    def showHelp(self):
        help_text = (
            "Usage Instructions:\n\n"
            "1. Select the input method: Single Sequence or File Input.\n"
            "2. Select the classification task:\n"
            "   - Nucleic acid-binding proteins (NABPs) / Non-nucleic acid-binding proteins (Non-NABPs): Binary classification distinguishing NABPs from non-NABPs.\n"
            "   - DNA-binding proteins (DBPs) / RNA-binding proteins (RBPs): Fine-grained classification differentiating DBPs and RBPs within NABPs.\n"
            "3. For Single Sequence:\n"
            "   - Enter a protein sequence directly into the text box.\n"
            "   - Click 'Predict' to see the result immediately on the interface.\n"
            "   - You can copy the result to clipboard by clicking 'Copy Result'.\n"
            "4. For File Input:\n"
            "   - Click 'Browse' to select a file. Supported formats:\n"
            "     - FASTA file (.fasta, .fa): Standard FASTA format with description lines starting with '>'.\n"
            "     - TXT file (.txt): Each line contains a protein sequence.\n"
            "     - TSV file (.tsv): Must contain a 'Sequence' column.\n"
            "   - Click 'Select' next to Output File Path to choose the output CSV file's name and location.\n"
            "   - Click 'Predict' to start batch prediction. Progress will be shown via the progress bar.\n"
            "   - Results will be saved to the specified CSV file.\n"
            "5. Ensure that all sequences contain only valid amino acid characters (ACDEFGHIKLMNPQRSTVWY).\n"
            "6. The maximum number of sequences for file input is 1000.\n\n"
            "Warnings:\n"
            "- Sequences with length less than 50 or greater than 1000 may result in less reliable predictions.\n\n"
            "For any issues, please check the 'drbp_edp.log' file for error details."
        )
        QMessageBox.information(self, "Help", help_text)

def main():
    app = QApplication(sys.argv)

    # 设置窗口图标（可选）
    icon_path = os.path.join(application_path, "icon.png")  # 确保icon.png位于应用程序目录
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        logging.error(f"Icon file not found at {icon_path}")

    # 创建主窗口但不显示
    main_window = MainWindow()
    main_window.hide()

    # 创建并显示自定义启动窗口
    splash = CustomSplashScreen(width=800, height=600)
    splash.show()
    QApplication.processEvents()

    # 启动模型加载，传递启动窗口和主窗口实例
    main_window.loadModels(splash, main_window)

    # 保持应用运行
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
