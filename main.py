# -*- coding: utf-8 -*-
"""
    This file is part of the PyGameGrounds project.
    Copyright (C) 2015-2016  <NAME>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import os
import re
import json
import time
import threading
import traceback
import logging
from logging.handlers import RotatingFileHandler
from PySide import QtGui, QtCore
from PySide.QtGui import *
from PySide.QtCore import *
from PySide.QtWidgets import *
from PySide.QtNetwork import *
from PySide.QtOpenGL import *

# ensure logs directory
os.makedirs("logs", exist_ok=True)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler = RotatingFileHandler("logs/run.log", maxBytes=5*1024*1024, backupCount=5)
handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# Optionally also log to console
ch = logging.StreamHandler()
ch.setFormatter(log_formatter)
logger.addHandler(ch)

logger.info("Application starting")

...