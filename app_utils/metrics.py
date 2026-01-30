# utils/metrics.py
# Responsibility: Track FPS and memory usage in real time

import time
import psutil

class Monitor:
    def __init__(self):
        """
        Initializes a performance monitor.
        Tracks total frames processed and time elapsed since start.
        """
        self.start = time.time()
        self.count = 0

    def update(self):
        """
        Call this once per processed frame.

        Returns:
            fps: frames per second since start
            mem: memory used (MB)
        """
        self.count += 1
        elapsed = time.time() - self.start
        fps = self.count / elapsed if elapsed > 0 else 0
        mem = psutil.virtual_memory().used / (1024 ** 2)  # MB
        return fps, mem



