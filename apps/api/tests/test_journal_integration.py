from unittest.mock import patch
from uuid import UUID

from app.models import JournalEntry, StateObservation


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


def test_authenticated_user_can_create_and_read_journal(client):
    register_response = register_user(
        client,
        "alice@example.com",
    )

    assert register_response.status_code == 201

    login_response = login_user(
        client,
        "alice@example.com",
    )

    assert login_response.status_code == 200

    with patch(
        "app.api.journals.process_journal.delay"
    ) as mocked_delay:
        create_response = client.post(
            "/v1/journals",
            json={
                "entry_type": "TEXT",
                "text": "Today was a productive day.",
            },
        )

    assert create_response.status_code == 202

    created = create_response.json()

    assert created["status"] == "QUEUED"
    assert "id" in created

    mocked_delay.assert_called_once_with(created["id"])

    get_response = client.get(
        f"/v1/journals/{created['id']}"
    )

    assert get_response.status_code == 200

    journal = get_response.json()

    assert journal["id"] == created["id"]
    assert journal["entry_type"] == "TEXT"
    assert journal["text"] == "Today was a productive day."
    assert journal["status"] == "QUEUED"
    assert journal["state"] is None
    
def test_anonymous_user_cannot_create_journal(client):
    response = client.post(
        "/v1/journals",
        json={
            "entry_type": "TEXT",
            "text": "This should not be accepted.",
        },
    )

    assert response.status_code == 401
    
def test_anonymous_user_cannot_list_journals(client):
    response = client.get("/v1/journals")

    assert response.status_code == 401
    
def test_users_cannot_access_each_others_journals(client):
    # Alice
    assert register_user(
        client,
        "alice@example.com",
        display_name="Alice",
    ).status_code == 201

    assert login_user(
        client,
        "alice@example.com",
    ).status_code == 200

    with patch("app.api.journals.process_journal.delay"):
        alice_create = client.post(
            "/v1/journals",
            json={
                "entry_type": "TEXT",
                "text": "Alice private journal",
            },
        )

    assert alice_create.status_code == 202
    alice_id = alice_create.json()["id"]

    # Log Alice out before switching users.
    assert client.post("/v1/auth/logout").status_code == 200

    # Bob
    assert register_user(
        client,
        "bob@example.com",
        display_name="Bob",
    ).status_code == 201

    assert login_user(
        client,
        "bob@example.com",
    ).status_code == 200

    with patch("app.api.journals.process_journal.delay"):
        bob_create = client.post(
            "/v1/journals",
            json={
                "entry_type": "TEXT",
                "text": "Bob private journal",
            },
        )

    assert bob_create.status_code == 202
    bob_id = bob_create.json()["id"]

    # Bob can access Bob's journal.
    response = client.get(f"/v1/journals/{bob_id}")
    assert response.status_code == 200

    # Bob must NOT be able to access Alice's journal.
    response = client.get(f"/v1/journals/{alice_id}")
    assert response.status_code == 404

    # Bob's history must contain only Bob's journal.
    response = client.get("/v1/journals")

    assert response.status_code == 200

    body = response.json()

    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == bob_id
    assert body["items"][0]["text"] == "Bob private journal"

    # Switch back to Alice.
    assert client.post("/v1/auth/logout").status_code == 200

    assert login_user(
        client,
        "alice@example.com",
    ).status_code == 200

    # Alice can access Alice's journal.
    response = client.get(f"/v1/journals/{alice_id}")
    assert response.status_code == 200

    # Alice must NOT be able to access Bob's journal.
    response = client.get(f"/v1/journals/{bob_id}")
    assert response.status_code == 404

    # Alice's history must contain only Alice's journal.
    response = client.get("/v1/journals")

    assert response.status_code == 200

    body = response.json()

    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == alice_id
    assert body["items"][0]["text"] == "Alice private journal"
    
    

def test_user_cannot_update_another_users_journal(client):
    register_user(client, "alice@example.com")
    login_user(client, "alice@example.com")

    with patch("app.api.journals.process_journal.delay"):
        create_response = client.post(
            "/v1/journals",
            json={
                "entry_type": "TEXT",
                "text": "Alice original text",
            },
        )

    alice_id = create_response.json()["id"]

    client.post("/v1/auth/logout")

    register_user(client, "bob@example.com")
    login_user(client, "bob@example.com")

    with patch(
        "app.api.journals.process_journal.delay"
    ) as mocked_delay:
        response = client.patch(
            f"/v1/journals/{alice_id}",
            json={
                "text": "Bob tried to change Alice's journal",
            },
        )

    assert response.status_code == 404

    # Unauthorized mutation must not trigger processing.
    mocked_delay.assert_not_called()

    # Verify Alice's original data remains unchanged.
    client.post("/v1/auth/logout")
    login_user(client, "alice@example.com")

    response = client.get(f"/v1/journals/{alice_id}")

    assert response.status_code == 200
    assert response.json()["text"] == "Alice original text"
    
    
    
    
def test_user_cannot_delete_another_users_journal(client):
    register_user(client, "alice@example.com")
    login_user(client, "alice@example.com")

    with patch("app.api.journals.process_journal.delay"):
        create_response = client.post(
            "/v1/journals",
            json={
                "entry_type": "TEXT",
                "text": "Alice journal must survive",
            },
        )

    alice_id = create_response.json()["id"]

    client.post("/v1/auth/logout")

    register_user(client, "bob@example.com")
    login_user(client, "bob@example.com")

    response = client.delete(
        f"/v1/journals/{alice_id}"
    )

    assert response.status_code == 404

    # Verify the resource actually survived.
    client.post("/v1/auth/logout")
    login_user(client, "alice@example.com")

    response = client.get(
        f"/v1/journals/{alice_id}"
    )

    assert response.status_code == 200
    assert response.json()["text"] == "Alice journal must survive"
    
    
    
def test_owner_can_update_journal_and_reprocessing_is_triggered(client):
    register_user(client, "alice@example.com")
    login_user(client, "alice@example.com")

    with patch("app.api.journals.process_journal.delay"):
        create_response = client.post(
            "/v1/journals",
            json={
                "entry_type": "TEXT",
                "text": "Original journal text",
            },
        )

    journal_id = create_response.json()["id"]

    with patch(
        "app.api.journals.process_journal.delay"
    ) as mocked_delay:
        update_response = client.patch(
            f"/v1/journals/{journal_id}",
            json={
                "text": "Updated journal text",
            },
        )

    assert update_response.status_code == 200

    body = update_response.json()

    assert body["id"] == journal_id
    assert body["status"] == "QUEUED"

    mocked_delay.assert_called_once_with(journal_id)

    get_response = client.get(
        f"/v1/journals/{journal_id}"
    )

    assert get_response.status_code == 200

    journal = get_response.json()

    assert journal["text"] == "Updated journal text"
    assert journal["status"] == "QUEUED"
    assert journal["state"] is None
    
    
    
def test_owner_can_delete_journal(client):
    register_user(client, "alice@example.com")
    login_user(client, "alice@example.com")

    with patch("app.api.journals.process_journal.delay"):
        create_response = client.post(
            "/v1/journals",
            json={
                "entry_type": "TEXT",
                "text": "Journal that will be deleted",
            },
        )

    journal_id = create_response.json()["id"]

    delete_response = client.delete(
        f"/v1/journals/{journal_id}"
    )

    assert delete_response.status_code == 204

    get_response = client.get(
        f"/v1/journals/{journal_id}"
    )

    assert get_response.status_code == 404

    list_response = client.get("/v1/journals")

    assert list_response.status_code == 200
    assert list_response.json()["total"] == 0
    
    
    
def test_empty_journal_text_is_rejected(client):
    register_user(client, "alice@example.com")
    login_user(client, "alice@example.com")

    with patch(
        "app.api.journals.process_journal.delay"
    ) as mocked_delay:
        response = client.post(
            "/v1/journals",
            json={
                "entry_type": "TEXT",
                "text": "",
            },
        )

    assert response.status_code == 422
    mocked_delay.assert_not_called()


def test_invalid_entry_type_is_rejected(client):
    register_user(client, "alice@example.com")
    login_user(client, "alice@example.com")

    with patch(
        "app.api.journals.process_journal.delay"
    ) as mocked_delay:
        response = client.post(
            "/v1/journals",
            json={
                "entry_type": "AUDIO",
                "text": "This should fail in M1.",
            },
        )

    assert response.status_code == 422
    mocked_delay.assert_not_called()


def test_empty_update_is_rejected(client):
    register_user(client, "alice@example.com")
    login_user(client, "alice@example.com")

    with patch("app.api.journals.process_journal.delay"):
        create_response = client.post(
            "/v1/journals",
            json={
                "entry_type": "TEXT",
                "text": "Original text",
            },
        )

    journal_id = create_response.json()["id"]

    with patch(
        "app.api.journals.process_journal.delay"
    ) as mocked_delay:
        response = client.patch(
            f"/v1/journals/{journal_id}",
            json={"text": ""},
        )

    assert response.status_code == 422
    mocked_delay.assert_not_called()

    # Validation failure must not mutate the original resource.
    response = client.get(f"/v1/journals/{journal_id}")

    assert response.status_code == 200
    assert response.json()["text"] == "Original text"
    
    
    
def test_journal_history_pagination(client):
    register_user(client, "alice@example.com")
    login_user(client, "alice@example.com")

    with patch("app.api.journals.process_journal.delay"):
        for number in range(5):
            response = client.post(
                "/v1/journals",
                json={
                    "entry_type": "TEXT",
                    "text": f"Journal {number}",
                },
            )

            assert response.status_code == 202

    response = client.get(
        "/v1/journals?limit=2&offset=0"
    )

    assert response.status_code == 200

    first_page = response.json()

    assert first_page["total"] == 5
    assert first_page["limit"] == 2
    assert first_page["offset"] == 0
    assert len(first_page["items"]) == 2

    response = client.get(
        "/v1/journals?limit=2&offset=2"
    )

    assert response.status_code == 200

    second_page = response.json()

    assert second_page["total"] == 5
    assert second_page["limit"] == 2
    assert second_page["offset"] == 2
    assert len(second_page["items"]) == 2

    first_ids = {
        item["id"]
        for item in first_page["items"]
    }

    second_ids = {
        item["id"]
        for item in second_page["items"]
    }

    assert first_ids.isdisjoint(second_ids)
    
    
    
def test_journal_pagination_limits_are_clamped(client):
    register_user(client, "alice@example.com")
    login_user(client, "alice@example.com")

    response = client.get(
        "/v1/journals?limit=500&offset=-50"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["limit"] == 100
    assert body["offset"] == 0
    
    
    
def test_deleting_journal_cascades_state_observation(
    client,
    db_session,
):
    register_user(client, "alice@example.com")
    login_user(client, "alice@example.com")

    with patch("app.api.journals.process_journal.delay"):
        create_response = client.post(
            "/v1/journals",
            json={
                "entry_type": "TEXT",
                "text": "Journal with persisted state",
            },
        )

    journal_id = UUID(create_response.json()["id"])

    journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert journal is not None

    observation = StateObservation(
        journal_id=journal_id,
        energy=0.75,
        stress=0.25,
        confidence=0.82,
        model_version="test-model",
    )

    db_session.add(observation)
    db_session.commit()

    observation_id = observation.id

    # Make sure the state really exists first.
    assert (
        db_session.get(StateObservation, observation_id)
        is not None
    )

    delete_response = client.delete(
        f"/v1/journals/{journal_id}"
    )

    assert delete_response.status_code == 204

    # Expire the session so we're checking the DB,
    # not stale SQLAlchemy identity-map state.
    db_session.expire_all()

    assert (
        db_session.get(StateObservation, observation_id)
        is None
    )