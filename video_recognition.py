import threading
import time

class VideoRecognitionModel:
    def __init__(self):
        self.paused = False
        self.pause_event = threading.Event()

    def pause(self):
        self.paused = True
        self.pause_event.set()
        logger.info("Paused inference; dialog open")

    def resume(self):
        self.paused = False
        self.pause_event.clear()
        logger.info("Resumed inference; dialog closed")

    def process_frame(self, frame):
        if self.paused:
            return None
        # ...existing code for frame processing...

def inference_loop():
    while True:
        if pause_event.is_set():
            time.sleep(0.1)
            continue
        # run detection
        # logger.info("Detection run started")
        # ...existing code...

def open_teach_dialog():
    # ...existing code...
    pause_event.set()
    logger.info("Teach dialog opened; paused inference")

def close_teach_dialog():
    # ...existing code...
    pause_event.clear()
    logger.info("Teach dialog closed; resumed inference")