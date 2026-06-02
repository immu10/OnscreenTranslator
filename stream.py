import dxcam


class Stream:
    def __init__(self, monitor=0, target_fps=60, color="BGR",
                 crop_top_ratio=0.0, crop_bottom_ratio=0.0):
        self.monitor = monitor
        self.target_fps = target_fps
        self.color = color
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.region = None  # (x1, y1, x2, y2) in monitor-local coords; set in start()
        self._camera = None

    def start(self):
        self._camera = dxcam.create(output_idx=self.monitor, output_color=self.color)
        w, h = self._camera.width, self._camera.height
        if self.crop_top_ratio > 0 or self.crop_bottom_ratio > 0:
            y_top = int(h * self.crop_top_ratio)
            y_bot = h - int(h * self.crop_bottom_ratio)
            self.region = (0, y_top, w, y_bot)
            print(f"[stream] cropping top {self.crop_top_ratio:.0%} "
                  f"bottom {self.crop_bottom_ratio:.0%} -> region={self.region}",
                  flush=True)
        else:
            self.region = (0, 0, w, h)
        self._camera.start(target_fps=self.target_fps, video_mode=True, region=self.region)

    def stop(self):
        if self._camera is not None:
            self._camera.stop()
            self._camera = None

    def get_frame(self):
        return self._camera.get_latest_frame()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
