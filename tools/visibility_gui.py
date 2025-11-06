from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

from road_visibility.config import PipelineConfig
from road_visibility.detectors import LazyUltralyticsVehicleDetector
from road_visibility.types import FrameVisibility
from road_visibility.visibility import RoadVisibilityEstimator


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray
    visibility: FrameVisibility


class VisibilityProcessorThread(threading.Thread):
    def __init__(
        self,
        video_path: Path,
        config: PipelineConfig,
        warmup_fraction: float,
        frame_stride: int,
        use_yolo: bool,
        yolo_weights: str,
        result_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.video_path = video_path
        self.config = config
        self.warmup_fraction = warmup_fraction
        self.frame_stride = max(1, frame_stride)
        self.use_yolo = use_yolo
        self.yolo_weights = yolo_weights
        self.result_queue = result_queue
        self.stop_event = stop_event

    def run(self) -> None:
        try:
            self._run_impl()
        except Exception as exc:  # pylint: disable=broad-except
            self._post(("error", str(exc)))

    def _run_impl(self) -> None:
        detectors = []
        if self.use_yolo:
            detectors.append(LazyUltralyticsVehicleDetector(self.config, model_name=self.yolo_weights))

        estimator = RoadVisibilityEstimator(config=self.config, vehicle_detectors=detectors)
        estimator.use_transmittance_video_fusion = True

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 1e-3:
                fps = 30.0

            warmup_frames: list[np.ndarray] = []
            accum: Optional[np.ndarray] = None

            warmup_target = self._compute_warmup_target(total_frames, fps)
            if warmup_target > 0:
                self._post(("status", f"Warm-up: 0 / {warmup_target} frames"))

            for idx in range(warmup_target):
                if self.stop_event.is_set():
                    return
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self._resize_for_estimator(frame, estimator)
                warmup_frames.append(frame)
                frame_float = frame.astype(np.float32)
                accum = frame_float if accum is None else accum + frame_float
                if idx and idx % 20 == 0:
                    self._post(("status", f"Warm-up: {idx + 1} / {warmup_target} frames"))

            if not warmup_frames:
                raise ValueError("Video contains no frames for warm-up.")

            averaged = (accum / max(len(warmup_frames), 1)).astype(np.uint8)
            estimator.warmup(averaged, additional_frames=warmup_frames)
            self._post(("status", "Warm-up complete. Starting visibility estimation..."))

            frames_consumed = len(warmup_frames)
            start_time = time.time()
            processed_frames = 0

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self._resize_for_estimator(frame, estimator)
                frames_consumed += 1

                if self.frame_stride > 1 and ((frames_consumed - len(warmup_frames)) % self.frame_stride != 0):
                    continue

                estimate = estimator.estimate(frame)
                frame_index = frames_consumed - 1
                timestamp = frame_index / fps
                packet = FramePacket(
                    frame=frame,
                    visibility=FrameVisibility(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        estimate=estimate,
                    ),
                )
                self._post(("frame", packet))
                processed_frames += 1

                if processed_frames % 30 == 0:
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        rate = processed_frames / elapsed
                        self._post(("status", f"Processing... {rate:.1f} fps"))

            self._post(("completed", None))
        finally:
            cap.release()

    def _compute_warmup_target(self, total_frames: int, fps: float) -> int:
        if total_frames > 0 and self.warmup_fraction > 0:
            target = int(total_frames * self.warmup_fraction)
        else:
            target = int(round(fps * 60.0))
        target = max(target, 1)
        if total_frames > 0:
            target = min(target, total_frames)
        return target

    def _resize_for_estimator(self, frame: np.ndarray, estimator: RoadVisibilityEstimator) -> np.ndarray:
        target_w = max(int(estimator.config.frame_target_width), 0)
        target_h = max(int(estimator.config.frame_target_height), 0)
        if target_w > 0 and target_h > 0 and (frame.shape[1] != target_w or frame.shape[0] != target_h):
            return cv2.resize(frame, (target_w, target_h))
        return frame

    def _post(self, item: Tuple[str, object]) -> None:
        try:
            self.result_queue.put_nowait(item)
        except queue.Full:
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
            self.result_queue.put_nowait(item)


class VisibilityApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Road Visibility Estimation")

        self.config = PipelineConfig()

        self.video_path: Optional[Path] = None
        self.processor_thread: Optional[VisibilityProcessorThread] = None
        self.queue: queue.Queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        self.current_photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_queue)

    def _build_ui(self) -> None:
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(control_frame, text="Open Video...", command=self._open_video).pack(side=tk.LEFT)

        self.video_label_var = tk.StringVar(value="No video selected")
        tk.Label(control_frame, textvariable=self.video_label_var, anchor="w").pack(side=tk.LEFT, padx=10)

        self.use_yolo_var = tk.BooleanVar(value=False)
        tk.Checkbutton(control_frame, text="Use YOLO", variable=self.use_yolo_var, command=self._toggle_yolo).pack(
            side=tk.LEFT, padx=10
        )

        self.yolo_weights_var = tk.StringVar(value="checkpoints/yolov8n.pt")
        self.yolo_entry = tk.Entry(control_frame, width=30, textvariable=self.yolo_weights_var, state=tk.DISABLED)
        self.yolo_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Frame stride:").pack(side=tk.LEFT, padx=(10, 2))
        self.frame_stride_var = tk.StringVar(value="1")
        tk.Entry(control_frame, width=5, textvariable=self.frame_stride_var).pack(side=tk.LEFT)

        tk.Button(control_frame, text="Start Detection", command=self._start).pack(side=tk.LEFT, padx=(10, 2))
        tk.Button(control_frame, text="Stop", command=self._stop).pack(side=tk.LEFT)

        self.video_panel = tk.Label(self.root, bd=2, relief=tk.SUNKEN)
        self.video_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        results_frame = tk.Frame(self.root)
        results_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.status_var = tk.StringVar(value="Select a video to begin.")
        tk.Label(results_frame, textvariable=self.status_var, anchor="w").pack(fill=tk.X)

        self.compare_var = tk.StringVar(value="Visibility (edge compare): -- m")
        self.trans_var = tk.StringVar(value="Visibility (transmittance): -- m")
        self.fused_var = tk.StringVar(value="Visibility (fused): -- m")

        tk.Label(results_frame, textvariable=self.compare_var, anchor="w").pack(fill=tk.X)
        tk.Label(results_frame, textvariable=self.trans_var, anchor="w").pack(fill=tk.X)
        tk.Label(results_frame, textvariable=self.fused_var, anchor="w").pack(fill=tk.X)

    def _toggle_yolo(self) -> None:
        state = tk.NORMAL if self.use_yolo_var.get() else tk.DISABLED
        self.yolo_entry.configure(state=state)

    def _open_video(self) -> None:
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
            ("All files", "*.*"),
        ]
        selected = filedialog.askopenfilename(title="Select video", filetypes=filetypes)
        if not selected:
            return
        self.video_path = Path(selected)
        self.video_label_var.set(str(self.video_path))
        self.status_var.set("Video selected. Ready to start detection.")

    def _validate_stride(self) -> int:
        try:
            stride = int(self.frame_stride_var.get())
        except ValueError:
            raise ValueError("Frame stride must be a positive integer.") from None
        if stride < 1:
            raise ValueError("Frame stride must be at least 1.")
        return stride

    def _start(self) -> None:
        if self.processor_thread and self.processor_thread.is_alive():
            messagebox.showinfo("Processing", "Detection already in progress.")
            return
        if self.video_path is None:
            messagebox.showwarning("Missing video", "Please select a video file first.")
            return

        try:
            stride = self._validate_stride()
        except ValueError as exc:
            messagebox.showerror("Invalid stride", str(exc))
            return

        self.stop_event.clear()
        self.queue = queue.Queue(maxsize=5)

        use_yolo = self.use_yolo_var.get()
        weights = self.yolo_weights_var.get()

        self.processor_thread = VisibilityProcessorThread(
            video_path=self.video_path,
            config=self.config,
            warmup_fraction=0.0,
            frame_stride=stride,
            use_yolo=use_yolo,
            yolo_weights=weights,
            result_queue=self.queue,
            stop_event=self.stop_event,
        )
        self.processor_thread.start()
        self.status_var.set("Starting detection...")
        self._reset_visibility_labels()

    def _stop(self) -> None:
        if self.processor_thread and self.processor_thread.is_alive():
            self.stop_event.set()
            self.status_var.set("Stopping...")

    def _reset_visibility_labels(self) -> None:
        self.compare_var.set("Visibility (edge compare): -- m")
        self.trans_var.set("Visibility (transmittance): -- m")
        self.fused_var.set("Visibility (fused): -- m")

    def _poll_queue(self) -> None:
        try:
            while True:
                message, payload = self.queue.get_nowait()
                if message == "frame":
                    self._handle_frame(payload)
                elif message == "status":
                    self.status_var.set(str(payload))
                elif message == "error":
                    self.status_var.set("Error encountered.")
                    messagebox.showerror("Processing error", str(payload))
                    self.stop_event.set()
                elif message == "completed":
                    if not self.stop_event.is_set():
                        self.status_var.set("Processing completed.")
                    else:
                        self.status_var.set("Processing stopped.")
        except queue.Empty:
            pass

        self.root.after(50, self._poll_queue)

    def _handle_frame(self, packet: FramePacket) -> None:
        frame_rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = self._scale_image(image)
        self.current_photo = ImageTk.PhotoImage(image=image)
        self.video_panel.configure(image=self.current_photo)

        est = packet.visibility.estimate
        compare_val = est.visibility_compare if est.visibility_compare is not None else float("nan")
        trans_val = est.visibility_transmittance
        fused_val = est.visibility_fused

        self.compare_var.set(f"Visibility (edge compare): {compare_val:.1f} m" if np.isfinite(compare_val) else "Visibility (edge compare): -- m")
        self.trans_var.set(
            f"Visibility (transmittance): {trans_val:.1f} m" if trans_val is not None else "Visibility (transmittance): -- m"
        )
        self.fused_var.set(
            f"Visibility (fused): {fused_val:.1f} m" if fused_val is not None else "Visibility (fused): -- m"
        )

    def _scale_image(self, image: Image.Image) -> Image.Image:
        max_width = 960
        max_height = 540
        width, height = image.size
        scale = min(max_width / width, max_height / height, 1.0)
        if scale < 1.0:
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def _on_close(self) -> None:
        self._stop()
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = VisibilityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
