import pytest
import numpy as np
from pathlib import Path

from casadei.media import VideoMedia


class TestVideoMediaPathBased:
    def test_create_from_path(self, tmp_path):
        path = tmp_path / "video.mp4"
        path.touch()
        media = VideoMedia(path=path)
        assert media.path == path
        assert media.frames is None
        assert media.fps == 16
        assert media.format == "mp4"

    def test_nonexistent_path_without_frames_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            VideoMedia(path=Path("/nonexistent/video.mp4"))

    def test_nonexistent_path_with_frames_ok(self):
        """A nonexistent path is fine if frames are provided (save-later pattern)."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        media = VideoMedia(path=Path("/future/output.mp4"), frames=frames)
        assert media.path == Path("/future/output.mp4")
        assert media.frame_count == 1


class TestVideoMediaFramesBased:
    def test_create_from_frames(self):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]
        media = VideoMedia.from_frames(frames, fps=24)
        assert media.frames is not None
        assert media.path is None
        assert media.fps == 24
        assert media.frame_count == 10

    def test_from_frames_default_fps(self):
        frames = [np.zeros((50, 50, 3), dtype=np.uint8)]
        media = VideoMedia.from_frames(frames)
        assert media.fps == 16

    def test_requires_path_or_frames(self):
        with pytest.raises(ValueError, match="at least one"):
            VideoMedia()


class TestVideoMediaProperties:
    def test_frame_count(self):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        media = VideoMedia.from_frames(frames)
        assert media.frame_count == 5

    def test_frame_count_no_frames(self, tmp_path):
        path = tmp_path / "video.mp4"
        path.touch()
        media = VideoMedia(path=path)
        assert media.frame_count == 0

    def test_duration_seconds(self):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(32)]
        media = VideoMedia.from_frames(frames, fps=16)
        assert media.duration_seconds == 2.0

    def test_duration_seconds_no_frames(self, tmp_path):
        path = tmp_path / "video.mp4"
        path.touch()
        media = VideoMedia(path=path)
        assert media.duration_seconds == 0.0


class TestVideoMediaSave:
    def test_save_without_frames_raises(self, tmp_path):
        path = tmp_path / "video.mp4"
        path.touch()
        media = VideoMedia(path=path)
        with pytest.raises(ValueError, match="No in-memory frames"):
            media.save(tmp_path / "output.mp4")
