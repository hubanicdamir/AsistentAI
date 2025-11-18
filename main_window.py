import sys
import os
import re
import json
import time
import threading
import queue
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as data
import matplotlib.pyplot as plt
import random
import copy
import warnings
warnings.filterwarnings("ignore")

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QInputDialog, QMessageBox,
    QProgressBar, QComboBox, QSlider, QFrame
)
from PyQt5.QtGui import (
    QPen, QBrush, QColor, QPolygon, QTransform
)
from PyQt5.QtCore import (
    QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup,
    QTime, QMetaObject, QRunnable, QProcess, QUrl
)
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, QProgressBar, QComboBox, QSlider, QFrame
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygon, QTransform
from PyQt5.QtCore import QThread, QEvent, QPropertyAnimation, QParallelAnimationGroup, QTime, QMetaObject, QRunnable, QProcess, QUrl
from PyQt5.QtWidgets import QInputDialog, Q