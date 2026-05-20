import dxcam


class Stream:
    def __init__(self, monitor=0, target_fps=60, color="BGR"):
        self.monitor = monitor
        self.target_fps = target_fps
        self.color = color
        self._camera = None

    def start(self):
        self._camera = dxcam.create(output_idx=self.monitor, output_color=self.color)
        self._camera.start(target_fps=self.target_fps, video_mode=True)

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
