from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from app.models import JournalAudio, JournalEntry
from app.services.storage.local import LocalAudioStorage


TEST_AUDIO_BYTES = b"fake-webm-audio-content"


def register_user(
    client,
    email: str,
    password: str = "password123",
    display_name: str = "Test User",
):
    return client.post(
        "/v1/auth/register",
        json={
            "email": email,
            "password": password,
            "display_name": display_name,
        },
    )


def login_user(
    client,
    email: str,
    password: str = "password123",
):
    return client.post(
        "/v1/auth/login",
        json={
            "email": email,
            "password": password,
        },
    )


def authenticated_user(client, email="alice@example.com"):
    register_response = register_user(client, email)
    assert register_response.status_code == 201

    login_response = login_user(client, email)
    assert login_response.status_code == 200


def upload_voice_journal(
    client,
    filename: str = "recording.webm",
    content: bytes = TEST_AUDIO_BYTES,
    content_type: str = "audio/webm",
):
    with patch(
        "app.api.journals.transcribe_journal_audio.delay"
    ):
        return client.post(
            "/v1/journals/voice",
            files={
                "file": (
                    filename,
                    content,
                    content_type,
                )
            },
        )

def test_authenticated_user_can_upload_voice_journal(
    client,
    db_session,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        response = upload_voice_journal(client)

    assert response.status_code == 202

    body = response.json()

    assert body["status"] == "QUEUED"
    assert "id" in body

    journal_id = UUID(body["id"])

    db_session.expire_all()

    journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert journal is not None
    assert journal.entry_type == "VOICE"
    assert journal.raw_text is None
    assert journal.status == "QUEUED"

    audio = journal.audio

    assert audio is not None
    assert audio.original_filename == "recording.webm"
    assert audio.mime_type == "audio/webm"
    assert audio.size_bytes == len(TEST_AUDIO_BYTES)
    assert audio.duration_seconds is None
    assert audio.transcription_status == "TRANSCRIPTION_QUEUED"

    assert storage.exists(audio.storage_key)
    assert storage.get(audio.storage_key) == TEST_AUDIO_BYTES

def test_voice_upload_accepts_parameterized_webm_mime_type(
    client,
    db_session,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        response = upload_voice_journal(
            client,
            filename="recording.webm",
            content=TEST_AUDIO_BYTES,
            content_type="audio/webm;codecs=opus",
        )

    assert response.status_code == 202

    journal_id = UUID(response.json()["id"])

    db_session.expire_all()

    journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert journal is not None
    assert journal.entry_type == "VOICE"
    assert journal.status == "QUEUED"

    audio = journal.audio

    assert audio is not None

    # Preserve the browser-reported MIME type in metadata.
    assert audio.mime_type == "audio/webm;codecs=opus"

    # But storage should use the normalized base MIME type
    # to determine the safe server-controlled extension.
    assert audio.storage_key.endswith("/audio.webm")

    assert storage.exists(audio.storage_key)
    assert storage.get(audio.storage_key) == TEST_AUDIO_BYTES

def test_voice_upload_uses_server_generated_storage_key(
    client,
    db_session,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    dangerous_filename = "../../private-secret.webm"

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        response = upload_voice_journal(
            client,
            filename=dangerous_filename,
        )

    assert response.status_code == 202

    journal_id = UUID(response.json()["id"])

    db_session.expire_all()

    audio = db_session.query(JournalAudio).filter(
        JournalAudio.journal_id == journal_id
    ).one()

    assert ".." not in audio.storage_key
    assert dangerous_filename not in audio.storage_key

    assert audio.storage_key.startswith("users/")
    assert f"/journals/{journal_id}/" in audio.storage_key
    assert audio.storage_key.endswith("/audio.webm")

    assert storage.exists(audio.storage_key)


def test_anonymous_user_cannot_upload_voice_journal(client):
    response = upload_voice_journal(client)

    assert response.status_code == 401


def test_unsupported_audio_type_is_rejected(
    client,
    db_session,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        response = upload_voice_journal(
            client,
            filename="recording.ogg",
            content=b"ogg-content",
            content_type="audio/ogg",
        )

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported audio format"

    db_session.expire_all()

    assert db_session.query(JournalEntry).count() == 0
    assert db_session.query(JournalAudio).count() == 0

    assert list(Path(tmp_path).rglob("*")) == []


def test_empty_audio_is_rejected(
    client,
    db_session,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        response = upload_voice_journal(
            client,
            content=b"",
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Audio file is empty"

    db_session.expire_all()

    assert db_session.query(JournalEntry).count() == 0
    assert db_session.query(JournalAudio).count() == 0

    assert list(Path(tmp_path).rglob("*")) == []


def test_audio_over_25_mb_is_rejected(
    client,
    db_session,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    oversized_audio = b"x" * (25 * 1024 * 1024 + 1)

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        response = upload_voice_journal(
            client,
            content=oversized_audio,
        )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Audio file exceeds the 25 MB limit"
    )

    db_session.expire_all()

    assert db_session.query(JournalEntry).count() == 0
    assert db_session.query(JournalAudio).count() == 0

    assert list(Path(tmp_path).rglob("*")) == []


def test_get_voice_journal_returns_safe_audio_metadata(
    client,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        create_response = upload_voice_journal(client)

    assert create_response.status_code == 202

    journal_id = create_response.json()["id"]

    response = client.get(
        f"/v1/journals/{journal_id}"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["id"] == journal_id
    assert body["entry_type"] == "VOICE"
    assert body["text"] is None
    assert body["status"] == "QUEUED"
    assert body["state"] is None

    assert body["audio"] is not None

    audio = body["audio"]

    assert audio["original_filename"] == "recording.webm"
    assert audio["mime_type"] == "audio/webm"
    assert audio["size_bytes"] == len(TEST_AUDIO_BYTES)
    assert audio["duration_seconds"] is None
    assert (
        audio["transcription_status"]
        == "TRANSCRIPTION_QUEUED"
    )

    # Internal storage implementation must never leak
    # through the public API.
    assert "storage_key" not in audio


def test_list_journals_includes_voice_audio_metadata(
    client,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        create_response = upload_voice_journal(client)

    assert create_response.status_code == 202

    journal_id = create_response.json()["id"]

    response = client.get("/v1/journals")

    assert response.status_code == 200

    body = response.json()

    assert body["total"] == 1
    assert len(body["items"]) == 1

    journal = body["items"][0]

    assert journal["id"] == journal_id
    assert journal["entry_type"] == "VOICE"
    assert journal["text"] is None

    assert journal["audio"] is not None
    assert journal["audio"]["mime_type"] == "audio/webm"
    assert (
        journal["audio"]["transcription_status"]
        == "TRANSCRIPTION_QUEUED"
    )

    assert "storage_key" not in journal["audio"]


def test_voice_journal_cannot_be_edited_as_text(
    client,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        create_response = upload_voice_journal(client)

    assert create_response.status_code == 202

    journal_id = create_response.json()["id"]

    with patch(
        "app.api.journals.process_journal.delay"
    ) as mocked_delay:
        response = client.patch(
            f"/v1/journals/{journal_id}",
            json={
                "text": "Trying to replace the transcript manually.",
            },
        )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Only text journals can be edited"
    )

    mocked_delay.assert_not_called()

    get_response = client.get(
        f"/v1/journals/{journal_id}"
    )

    assert get_response.status_code == 200
    assert get_response.json()["text"] is None


def test_user_cannot_access_another_users_voice_journal(
    client,
    tmp_path,
):
    storage = LocalAudioStorage(str(tmp_path))

    # Alice creates private voice journal.
    authenticated_user(
        client,
        email="alice@example.com",
    )

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        create_response = upload_voice_journal(client)

    assert create_response.status_code == 202

    alice_journal_id = create_response.json()["id"]

    assert client.post(
        "/v1/auth/logout"
    ).status_code == 200

    # Bob signs in.
    authenticated_user(
        client,
        email="bob@example.com",
    )

    response = client.get(
        f"/v1/journals/{alice_journal_id}"
    )

    # Preserve the M1 anti-enumeration behaviour.
    assert response.status_code == 404

    list_response = client.get("/v1/journals")

    assert list_response.status_code == 200
    assert list_response.json()["total"] == 0


def test_database_failure_removes_stored_audio(
    client,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    with (
        patch(
            "app.api.journals.get_audio_storage",
            return_value=storage,
        ),
        patch(
            "sqlalchemy.orm.Session.commit",
            side_effect=RuntimeError("Simulated database failure"),
        ),
    ):
        try:
            upload_voice_journal(client)
        except RuntimeError:
            pass

    # A DB failure after storage.put() must not leave
    # private orphaned audio behind.
    stored_files = [
        path
        for path in Path(tmp_path).rglob("*")
        if path.is_file()
    ]

    assert stored_files == []
    
def test_deleting_voice_journal_removes_audio_file(
    client,
    db_session,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        create_response = upload_voice_journal(client)

    assert create_response.status_code == 202

    journal_id = UUID(create_response.json()["id"])

    db_session.expire_all()

    audio = db_session.query(JournalAudio).filter(
        JournalAudio.journal_id == journal_id
    ).one()

    audio_id = audio.id
    storage_key = audio.storage_key

    assert storage.exists(storage_key)

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        delete_response = client.delete(
            f"/v1/journals/{journal_id}"
        )

    assert delete_response.status_code == 204

    assert not storage.exists(storage_key)

    db_session.expire_all()

    assert db_session.get(
        JournalEntry,
        journal_id,
    ) is None

    assert db_session.get(
        JournalAudio,
        audio_id,
    ) is None


def test_deleting_text_journal_does_not_touch_audio_storage(
    client,
):
    authenticated_user(client)

    with patch(
        "app.api.journals.process_journal.delay"
    ):
        create_response = client.post(
            "/v1/journals",
            json={
                "entry_type": "TEXT",
                "text": "Normal text journal",
            },
        )

    assert create_response.status_code == 202

    journal_id = create_response.json()["id"]

    with patch(
        "app.api.journals.get_audio_storage"
    ) as mocked_storage:
        response = client.delete(
            f"/v1/journals/{journal_id}"
        )

    assert response.status_code == 204

    mocked_storage.assert_not_called()


def test_owner_can_retrieve_private_voice_audio(
    client,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        create_response = upload_voice_journal(client)

    assert create_response.status_code == 202

    journal_id = create_response.json()["id"]

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        response = client.get(
            f"/v1/journals/{journal_id}/audio"
        )

    assert response.status_code == 200

    assert response.content == TEST_AUDIO_BYTES

    assert response.headers[
        "content-type"
    ].startswith("audio/webm")

    assert response.headers[
        "cache-control"
    ] == "private, no-store"

    assert response.headers[
        "content-disposition"
    ] == "inline"
    
    
def test_anonymous_user_cannot_retrieve_voice_audio(
    client,
    tmp_path,
):
    authenticated_user(client)

    storage = LocalAudioStorage(str(tmp_path))

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        create_response = upload_voice_journal(client)

    assert create_response.status_code == 202

    journal_id = create_response.json()["id"]

    logout_response = client.post(
        "/v1/auth/logout"
    )

    assert logout_response.status_code == 200

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        response = client.get(
            f"/v1/journals/{journal_id}/audio"
        )

    assert response.status_code == 401
    

def test_user_cannot_retrieve_another_users_voice_audio(
    client,
    tmp_path,
):
    storage = LocalAudioStorage(str(tmp_path))

    authenticated_user(
        client,
        email="alice-audio@example.com",
    )

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ):
        create_response = upload_voice_journal(client)

    assert create_response.status_code == 202

    alice_journal_id = create_response.json()["id"]

    assert client.post(
        "/v1/auth/logout"
    ).status_code == 200

    authenticated_user(
        client,
        email="bob-audio@example.com",
    )

    with patch(
        "app.api.journals.get_audio_storage",
        return_value=storage,
    ) as mocked_storage:
        response = client.get(
            f"/v1/journals/{alice_journal_id}/audio"
        )

    assert response.status_code == 404

    # Ownership must be checked before private storage
    # is accessed.
    mocked_storage.assert_not_called()
    
def test_text_journal_has_no_audio_resource(
    client,
):
    authenticated_user(client)

    with patch(
        "app.api.journals.process_journal.delay"
    ):
        create_response = client.post(
            "/v1/journals",
            json={
                "entry_type": "TEXT",
                "text": "This journal has no audio.",
            },
        )

    assert create_response.status_code == 202

    journal_id = create_response.json()["id"]

    with patch(
        "app.api.journals.get_audio_storage"
    ) as mocked_storage:
        response = client.get(
            f"/v1/journals/{journal_id}/audio"
        )

    assert response.status_code == 404

    mocked_storage.assert_not_called()