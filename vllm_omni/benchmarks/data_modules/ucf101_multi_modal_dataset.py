import base64
import logging
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
from vllm.benchmarks.datasets import RandomMultiModalDataset

logger = logging.getLogger(__name__)


def load_ucf101_subset(
    dataset_path: str,
    subset_ratio: float = 0.1,
    random_seed: int = 42,
) -> list[str]:
    """
    Load a subset of the UCF101 dataset following the standard UCF101 directory structure.

    UCF101 directory format: dataset_path/Class_Name/Video_Name.avi/mp4

    Raises:
        ValueError: If no video files are found or the input path is invalid.
    """
    if not os.path.isdir(dataset_path):
        raise ValueError(f"UCF101 dataset path is not a valid directory: {dataset_path}")

    video_paths = []
    rng = np.random.RandomState(random_seed)

    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_videos = [
            os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith((".avi", ".mp4"))
        ]
        if not class_videos:
            logger.warning(f"No video files found in class directory: {class_dir}")
            continue

        subset_size = max(1, int(len(class_videos) * subset_ratio))
        subset_videos = rng.choice(class_videos, size=subset_size, replace=False).tolist()
        video_paths.extend(subset_videos)

    if not video_paths:
        raise ValueError(f"No valid UCF101 video files found in {dataset_path} (support .avi/.mp4)")

    logger.info(
        f"Successfully loaded UCF101 subset: {len(video_paths)} videos from {len(os.listdir(dataset_path))} classes"
    )
    return video_paths


def process_ucf101_video(video_file_path: str) -> Mapping[str, Any]:
    """
    Process a single UCF101 video file and return a multimedia content dictionary.

    Aligns with the output format of process_video from RandomMultiModalDataset to
    ensure upper-layer logic is unaware of the underlying video source.

    Raises:
        FileNotFoundError: If the video file does not exist.
        IOError: If reading the video file fails.
    """
    if not os.path.exists(video_file_path):
        raise FileNotFoundError(f"UCF101 video file not found: {video_file_path}")

    try:
        with open(video_file_path, "rb") as f:
            video_raw_bytes = f.read()
        video_base64 = base64.b64encode(video_raw_bytes).decode("utf-8")
    except Exception as e:
        raise OSError(f"Failed to read UCF101 video {video_file_path}: {str(e)}")

    return {
        "type": "video_url",
        "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
    }


# -----------------------------------------------------------------------------
# UCF101 MultiModalDataset Implementation
# -----------------------------------------------------------------------------
class UCF101MultiModalDataset(RandomMultiModalDataset):
    def __init__(
        self,
        dataset_path: str,
        subset_ratio: float = 0.1,
        random_seed: int = 42,
        **kwargs,
    ):
        super().__init__(random_seed=random_seed, **kwargs)

        self.dataset_path = dataset_path
        self.subset_ratio = subset_ratio

        self.ucf101_video_paths = load_ucf101_subset(
            dataset_path=dataset_path,
            subset_ratio=subset_ratio,
            random_seed=random_seed,
        )

        self._video_rng = np.random.RandomState(random_seed)

    def sample_ucf101_video(self) -> Mapping[str, Any]:
        selected_video_path = self._video_rng.choice(self.ucf101_video_paths)
        return process_ucf101_video(selected_video_path)

    def generate_mm_item(
        self,
        mm_item_config: tuple[int, int, int],
    ) -> Mapping[str, Any]:
        """
        Create UCF101 video items or synthetic image/audio items.

        Follows the OpenAI API chat completions format:
        https://github.com/openai/openai-python
        """

        modality = self.map_config_to_modality(mm_item_config)

        if modality == "video":
            return self.sample_ucf101_video()
        elif modality in ("image", "audio"):
            return super().generate_mm_item(mm_item_config)
        else:
            raise ValueError(f"Invalid multimodal item configuration: {mm_item_config}")
