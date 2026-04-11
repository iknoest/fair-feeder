"""Tests for roboflow_upload module."""
import io
import re
from unittest.mock import MagicMock, patch, call
from datetime import datetime

import pytest
from PIL import Image

from flagging import FlaggedFrame
from roboflow_upload import upload_flagged_frames, format_telegram_flag_summary, UploadResult


def _minimal_jpeg(width=64, height=64):
    """Return bytes of a minimal valid JPEG for testing."""
    buf = io.BytesIO()
    Image.new('RGB', (width, height), color=(128, 128, 128)).save(buf, format='JPEG')
    return buf.getvalue()


class TestUploadFlaggedFrames:

    def _make_frame(self, idx=0, tags=None, detections=None):
        return FlaggedFrame(
            frame_idx=idx,
            jpeg=_minimal_jpeg(),
            tags=tags or ['low-conf-sanbo-31'],
            max_conf=0.31,
            detections=detections or [],
        )

    @patch('roboflow_upload.Roboflow')
    def test_uploads_frame_to_roboflow(self, mock_rf_cls):
        mock_project = MagicMock()
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        ff = self._make_frame(idx=10)
        result = upload_flagged_frames(
            [ff], api_key='key', workspace='ws', project='proj', video_stem='clip1'
        )

        assert result.uploaded == 1
        assert result.failed == 0
        mock_project.upload.assert_called_once()

    @patch('roboflow_upload.Roboflow')
    def test_batch_name_uses_current_month(self, mock_rf_cls):
        mock_project = MagicMock()
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        ff = self._make_frame()
        upload_flagged_frames(
            [ff], api_key='key', workspace='ws', project='proj', video_stem='clip1'
        )

        _, kwargs = mock_project.upload.call_args
        expected = f"flagged-{datetime.now().strftime('%Y-%m')}"
        assert kwargs['batch_name'] == expected

    @patch('roboflow_upload.Roboflow')
    def test_filename_includes_video_stem_and_frame(self, mock_rf_cls):
        mock_project = MagicMock()
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        ff = self._make_frame(idx=42)
        upload_flagged_frames(
            [ff], api_key='key', workspace='ws', project='proj', video_stem='morning_feed'
        )

        _, kwargs = mock_project.upload.call_args
        path = kwargs['image_path']
        assert 'morning_feed' in path
        assert 'frame00042' in path
        assert path.endswith('.jpg')

    @patch('roboflow_upload.Roboflow')
    def test_handles_upload_failure_gracefully(self, mock_rf_cls):
        mock_project = MagicMock()
        mock_project.upload.side_effect = Exception("API error")
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        ff = self._make_frame()
        result = upload_flagged_frames(
            [ff], api_key='key', workspace='ws', project='proj', video_stem='clip1'
        )

        assert result.uploaded == 0
        assert result.failed == 1

    def test_empty_list_returns_zero(self):
        result = upload_flagged_frames(
            [], api_key='key', workspace='ws', project='proj', video_stem='clip1'
        )
        assert result.uploaded == 0
        assert result.failed == 0
        assert result.tag_counts == {}

    @patch('roboflow_upload.Roboflow')
    def test_tags_passed_to_roboflow(self, mock_rf_cls):
        mock_project = MagicMock()
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        ff = self._make_frame(tags=['low-conf-dan-25', 'blip-kibble'])
        upload_flagged_frames(
            [ff], api_key='key', workspace='ws', project='proj', video_stem='clip1'
        )

        _, kwargs = mock_project.upload.call_args
        assert kwargs['tag_names'] == ['low-conf-dan-25', 'blip-kibble']

    @patch('roboflow_upload.Roboflow')
    def test_tag_counts_strip_trailing_number(self, mock_rf_cls):
        mock_project = MagicMock()
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        frames = [
            self._make_frame(idx=0, tags=['low-conf-sanbo-31']),
            self._make_frame(idx=10, tags=['low-conf-sanbo-22', 'blip-dan']),
        ]
        result = upload_flagged_frames(
            frames, api_key='key', workspace='ws', project='proj', video_stem='clip1'
        )

        assert result.tag_counts['low-conf-sanbo'] == 2
        assert result.tag_counts['blip-dan'] == 1

    @patch('roboflow_upload.Roboflow')
    def test_custom_batch_name(self, mock_rf_cls):
        mock_project = MagicMock()
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        ff = self._make_frame()
        upload_flagged_frames(
            [ff], api_key='key', workspace='ws', project='proj',
            video_stem='clip1', batch_name='custom-batch'
        )

        _, kwargs = mock_project.upload.call_args
        assert kwargs['batch_name'] == 'custom-batch'


class TestFormatTelegramSummary:

    def test_zero_flags(self):
        result = UploadResult(uploaded=0, failed=0)
        assert format_telegram_flag_summary(result) == "No suspicious detections flagged"

    def test_all_uploaded(self):
        result = UploadResult(uploaded=3, failed=0, tag_counts={
            'low-conf-sanbo': 2, 'blip-dan': 1
        })
        text = format_telegram_flag_summary(result)
        assert "Auto-flagged: 3 frames -> Roboflow" in text
        assert "2x low-conf-sanbo" in text
        assert "1x blip-dan" in text
        # No failure count in output
        assert "failed" not in text.lower()

    def test_partial_failure(self):
        result = UploadResult(uploaded=2, failed=1, tag_counts={
            'low-conf-dan': 2, 'conflict-dan-sanbo': 1
        })
        text = format_telegram_flag_summary(result)
        assert "3 frames" in text
        assert "2 uploaded" in text
        assert "1 failed" in text

    def test_tags_sorted_by_frequency(self):
        result = UploadResult(uploaded=5, failed=0, tag_counts={
            'blip-dan': 1, 'low-conf-sanbo': 3, 'conflict-dan-sanbo': 2
        })
        text = format_telegram_flag_summary(result)
        # Find tag positions - most frequent first
        pos_sanbo = text.index('low-conf-sanbo')
        pos_conflict = text.index('conflict-dan-sanbo')
        pos_dan = text.index('blip-dan')
        assert pos_sanbo < pos_conflict < pos_dan
