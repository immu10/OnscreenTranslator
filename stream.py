import dxcam


class Stream:
    def __init__(self, monitor=0, target_fps=60, color="BGR",
                 crop_top_ratio=0.0, crop_bottom_ratio=0.0,
                 custom_region=None):
        self.monitor = monitor
        self.target_fps = target_fps
        self.color = color
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.custom_region = custom_region   # (x1, y1, x2, y2) monitor-local, or None
        self.region = None    # the region we actually used, populated in start()
        self._camera = None

    def _compute_region(self, w, h):
        """Decide the dxcam region from current settings.
        custom_region takes priority over crop ratios."""
        if self.custom_region is not None:
            rx1, ry1, rx2, ry2 = self.custom_region
            # clamp to monitor bounds in case the picked region went off-screen
            rx1 = max(0, min(rx1, w - 2))
            ry1 = max(0, min(ry1, h - 2))
            rx2 = max(rx1 + 1, min(rx2, w))
            ry2 = max(ry1 + 1, min(ry2, h))
            return (rx1, ry1, rx2, ry2)
        if self.crop_top_ratio > 0 or self.crop_bottom_ratio > 0:
            y_top = int(h * self.crop_top_ratio)
            y_bot = h - int(h * self.crop_bottom_ratio)
            return (0, y_top, w, y_bot)
        return (0, 0, w, h)

    def start(self):
        try:
            self._camera = dxcam.create(output_idx=self.monitor,
                                        output_color=self.color)
        except Exception as e:
            print(f"[stream] dxcam.create(output_idx={self.monitor}) failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            raise
        w, h = self._camera.width, self._camera.height
        self.region = self._compute_region(w, h)
        print(f"[stream] monitor={self.monitor} region={self.region}", flush=True)
        try:
            self._camera.start(target_fps=self.target_fps, video_mode=True,
                               region=self.region)
        except Exception as e:
            print(f"[stream] camera.start() failed: {type(e).__name__}: {e}",
                  flush=True)
            self._camera = None
            raise

    def stop(self):
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception as e:
                print(f"[stream] camera.stop() error: {e}", flush=True)
            # Drop our reference and try to release dxcam's internal cache so
            # that a subsequent create() with a different output_idx works.
            self._camera = None
            try:
                # dxcam keeps a module-level _camera reference and a device
                # cache. release() (recent versions) clears it; fall back to
                # poking the internals on older versions.
                if hasattr(dxcam, "reset"):
                    dxcam.reset()
                elif hasattr(dxcam, "_camera"):
                    dxcam._camera = None
            except Exception as e:
                print(f"[stream] dxcam cache clear failed: {e}", flush=True)

    def reconfigure(self, monitor=None, crop_top_ratio=None,
                    crop_bottom_ratio=None, custom_region=...):
        """Stop the current capture and restart with new parameters.
        Pass custom_region=None explicitly to clear it (the sentinel is used
        because None is itself a meaningful value for 'no custom region')."""
        self.stop()
        if monitor is not None:
            self.monitor = monitor
        if crop_top_ratio is not None:
            self.crop_top_ratio = crop_top_ratio
        if crop_bottom_ratio is not None:
            self.crop_bottom_ratio = crop_bottom_ratio
        if custom_region is not ...:
            self.custom_region = custom_region
        self.start()
        return self.region

    def get_frame(self):
        cam = self._camera  # snapshot to survive a swap mid-call
        if cam is None:
            return None
        try:
            return cam.get_latest_frame()
        except Exception:
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
