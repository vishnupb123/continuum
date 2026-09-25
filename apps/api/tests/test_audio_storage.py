import pytest

from app.services.storage.local import LocalAudioStorage


def test_put_get_exists_delete(tmp_path):
    storage = LocalAudioStorage(str(tmp_path))

    key = "users/user-123/journals/journal-456/audio.webm"
    content = b"continuum-test-audio"

    assert storage.exists(key) is False

    returned_key = storage.put(key, content)

    assert returned_key == key
    assert storage.exists(key) is True
    assert storage.get(key) == content

    storage.delete(key)

    assert storage.exists(key) is False


def test_nested_directories_are_created(tmp_path):
    storage = LocalAudioStorage(str(tmp_path))

    key = "users/user-1/journals/journal-1/audio.webm"

    storage.put(key, b"audio")

    assert storage.get(key) == b"audio"


def test_missing_object_raises_file_not_found(tmp_path):
    storage = LocalAudioStorage(str(tmp_path))

    with pytest.raises(FileNotFoundError):
        storage.get("does-not-exist.webm")


def test_path_traversal_is_rejected(tmp_path):
    storage = LocalAudioStorage(str(tmp_path))

    with pytest.raises(ValueError):
        storage.put("../../outside.webm", b"malicious")


def test_absolute_path_is_rejected(tmp_path):
    storage = LocalAudioStorage(str(tmp_path))

    with pytest.raises(ValueError):
        storage.put("/tmp/outside.webm", b"malicious")


def test_empty_key_is_rejected(tmp_path):
    storage = LocalAudioStorage(str(tmp_path))

    with pytest.raises(ValueError):
        storage.put("", b"audio")


def test_non_bytes_data_is_rejected(tmp_path):
    storage = LocalAudioStorage(str(tmp_path))

    with pytest.raises(TypeError):
        storage.put("audio.webm", "not-bytes")