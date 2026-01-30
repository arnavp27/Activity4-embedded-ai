# pipeline/sampler.py
# Responsibility: Allow only 1 frame every (1 / target_fps) seconds

import time

class FrameSampler:
    def __init__(self, target_fps=5):
        """
        Drops frames to ensure real-time performance.
        Example: If target_fps = 5, allow 1 frame every 0.2 seconds.
        """
        self.interval = 1.0 / target_fps
        self.last = 0  # timestamp of last allowed frame

    def allow(self):
        """
        Returns True if enough time has passed since last allowed frame.
        Otherwise, returns False (drop this frame).
        """
        now = time.time()
        if now - self.last >= self.interval:
            self.last = now
            return True
        return False
