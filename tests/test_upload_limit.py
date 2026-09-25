"""Tests for the admin's upload size limit."""

import json

import pytest

from src.data_pipeline.upload_limit import UploadLimit
from tests.conftest import make_settings

MB = 1024 * 1024


def test_defaults_to_the_configured_limit(tmp_path):
    limit = UploadLimit(default_mb=20, max_mb=200, store_path=tmp_path / "limit.json")
    assert limit.current_mb() == 20
    assert limit.max_bytes() == 20 * MB


def test_set_changes_the_limit_and_survives_a_restart(tmp_path):
    path = tmp_path / "limit.json"
    UploadLimit(20, 200, path).set_mb(50)

    assert UploadLimit(20, 200, path).current_mb() == 50
    assert json.loads(path.read_text(encoding="utf-8"))["max_mb"] == 50


def test_works_without_a_file():
    limit = UploadLimit(20, 200)
    limit.set_mb(5)
    assert limit.current_mb() == 5


@pytest.mark.parametrize("mb", [0, -1, 201])
def test_rejects_limits_out_of_range(mb):
    limit = UploadLimit(20, 200)
    with pytest.raises(ValueError, match="between 1 and 200 MB"):
        limit.set_mb(mb)
    assert limit.current_mb() == 20


def test_saved_limit_above_a_lowered_maximum_falls_back_to_default(tmp_path):
    path = tmp_path / "limit.json"
    UploadLimit(20, 200, path).set_mb(150)
    assert UploadLimit(20, 100, path).current_mb() == 20


@pytest.mark.parametrize("content", ["not json", '{"other": 1}', '{"max_mb": "big"}',
                                     '{"max_mb": true}'])
def test_broken_file_falls_back_to_default(tmp_path, content):
    path = tmp_path / "limit.json"
    path.write_text(content, encoding="utf-8")
    assert UploadLimit(20, 200, path).current_mb() == 20


def test_default_must_be_within_range():
    with pytest.raises(ValueError):
        UploadLimit(default_mb=300, max_mb=200)


def test_from_settings(tmp_path):
    settings = make_settings(
        max_upload_size_mb=10, max_upload_size_cap_mb=50, upload_limit_file=tmp_path / "l.json"
    )
    limit = UploadLimit.from_settings(settings)
    assert (limit.default_mb, limit.max_mb, limit.current_mb()) == (10, 50, 10)
