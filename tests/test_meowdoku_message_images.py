"""Per-chat consent, atomic PNG attachments and the real shared REST route."""
import base64
import http.client
import io
import json
import sqlite3

import pytest
from PIL import Image

from tests.test_meowdoku_messaging import server, _user, _friends, _rest_call, srv, messaging


def png(color="green"):
    stream = io.BytesIO()
    Image.new("RGB", (24, 16), color).save(stream, format="PNG")
    return stream.getvalue()


def encoded(raw):
    return {"data": base64.b64encode(raw).decode()}


@pytest.fixture
def db(tmp_path):
    connection = messaging.connect(tmp_path / "accounts.db")
    connection.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, username TEXT, display_name TEXT)")
    connection.executemany("INSERT INTO users VALUES (?,?,?)", [(1,"ann","Ann"),(2,"ben","Ben"),(3,"cyd","Cyd")])
    messaging.request_contact(connection, 1, "ben")
    messaging.respond_contact(connection, 2, "ann", True)
    yield connection
    connection.close()


def handle(db, me):
    return messaging.conversations(db, me)["contacts"][0]["conversation_handle"]


def enable(db, me, enabled=True):
    return messaging.image_consent(db, me, handle(db, me), enabled)


def send(db, data=None, key="image-operation-00001"):
    return messaging.send(db, 1, None, "Here is a Phong image", conversation_handle=handle(db, 1),
                          idempotency_key=key, image=encoded(data or png()))


def test_locked_then_same_message_unlocked_and_relocked(db):
    assert messaging.image_status(db, 1, "ben")["enabled"] is False
    with pytest.raises(messaging.MessagingError, match="enable images"):
        send(db)
    assert enable(db, 1)["available"] is False
    receipt = send(db)
    mid = receipt["id"]
    for me, peer in ((1,"ben"),(2,"ann")):
        assert messaging.thread(db, me, peer)["messages"][0]["image"] == {"locked": True}
        with pytest.raises(messaging.MessagingError):
            messaging.image_bytes(db, me, mid)
    assert enable(db, 2)["available"] is True
    for messages in (messaging.thread(db, 2, "ann")["messages"], messaging.since(db, 2, 0), messaging.inbox(db, 2)["messages"]):
        assert messages[0]["id"] == mid
        assert messages[0]["image"] == {"locked": False, "url": f"/api/messages/image?id={mid}", "mime":"image/png", "bytes":len(png())}
    assert messaging.image_bytes(db, 2, mid) == png()
    assert send(db) == receipt
    assert db.execute("SELECT count(*) FROM msg_images").fetchone()[0] == 1
    with pytest.raises(messaging.MessagingError, match="different content"):
        send(db, png("red"))
    enable(db, 2, False)
    with pytest.raises(messaging.MessagingError):
        messaging.image_bytes(db, 2, mid)
    assert messaging.thread(db, 1, "ben")["messages"][0]["image"] == {"locked": True}


def test_other_account_removed_message_and_block_regrant(db):
    enable(db, 1); enable(db, 2)
    mid = send(db)["id"]
    with pytest.raises(messaging.MessagingError):
        messaging.image_bytes(db, 3, mid)
    with pytest.raises(messaging.MessagingError):
        messaging.image_consent(db, 3, handle(db, 1), True)
    messaging.block(db, 2, "ann")
    with pytest.raises(messaging.MessagingError):
        messaging.image_bytes(db, 1, mid)
    messaging.request_contact(db, 2, "ann")
    assert messaging.image_status(db, 2, "ann")["enabled"] is False
    assert messaging.image_status(db, 1, "ben")["enabled"] is False
    enable(db, 1); enable(db, 2)
    messaging.admin_remove(db, 3, mid)
    assert "image" not in messaging.thread(db, 2, "ann")["messages"][0]
    with pytest.raises(messaging.MessagingError):
        messaging.image_bytes(db, 1, mid)


@pytest.mark.parametrize("value", [{"data":"no!"}, {"data":""}, {"url":"https://example.test/i.png"},
                                    {"data":base64.b64encode(b"<svg></svg>").decode()}, {"data":"A" * (2800000)}])
def test_invalid_images_never_create_message(db, value):
    enable(db, 1)
    with pytest.raises(messaging.MessagingError):
        messaging.send(db, 1, None, "test", conversation_handle=handle(db, 1), idempotency_key="invalid-image-00001", image=value)
    assert db.execute("SELECT count(*) FROM msg_messages").fetchone()[0] == 0


def test_truncated_dimensions_animation_and_quota(db, monkeypatch):
    enable(db, 1)
    for raw in (png()[:-10], b"\x89PNG\r\n\x1a\nwrong"):
        with pytest.raises(messaging.MessagingError):
            send(db, raw)
    stream = io.BytesIO()
    Image.new("RGB", (4097, 1)).save(stream, format="PNG")
    with pytest.raises(messaging.MessagingError):
        send(db, stream.getvalue())
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(stream, format="PNG", save_all=True, append_images=[Image.new("RGB", (2,2), "blue")])
    with pytest.raises(messaging.MessagingError):
        send(db, stream.getvalue())
    monkeypatch.setattr(messaging, "MAX_IMAGE_STORAGE_PER_ACCOUNT", len(png()) - 1)
    with pytest.raises(messaging.MessagingError, match="quota"):
        send(db)
    assert db.execute("SELECT count(*) FROM msg_messages").fetchone()[0] == 0


def test_consent_validation_and_legacy_text_shape(db):
    for bad in (1, "true", None):
        with pytest.raises(messaging.MessagingError):
            messaging.image_consent(db, 1, handle(db, 1), bad)
    with pytest.raises(messaging.MessagingError):
        messaging.send(db, 1, "ben", "hi", image=encoded(png()))
    messaging.send(db, 1, "ben", "normal text")
    assert "image" not in messaging.thread(db, 2, "ann")["messages"][0]


def test_http_unlock_bytes_and_access_gates(server, monkeypatch):
    monkeypatch.setattr(srv.proxy_trust_mod, "TRUSTED_PROXIES", {"127.0.0.1"})
    ca, cb, cc = _user(server, "ann"), _user(server, "ben"), _user(server, "cyd")
    _friends(server, "ann", ca, "ben", cb)
    ta, tb, tc = [c.split("=", 1)[1] for c in (ca, cb, cc)]
    def call(path, token=ta, body=None, secure=True):
        return _rest_call(server, path, token, secure=secure, body=body)
    ha = call("/api/messages/conversations")[1]["contacts"][0]["conversation_handle"]
    assert call("/api/messages/images?peer=ben", secure=False)[0] == 403
    assert call("/api/messages/images", body={"conversation_handle":ha,"enabled":True,"peer":"cyd"})[0] == 400
    assert call("/api/messages/images", body={"conversation_handle":ha,"enabled":True})[0] == 200
    payload = {"body":"Unlock this Phong image", "conversation_handle":ha, "idempotency_key":"http-image-00001", "image":encoded(png())}
    status, receipt = call("/api/messages/send", body=payload)
    assert status == 200
    url = f"/api/messages/image?id={receipt['id']}"
    assert call(url, tb)[0] == 403
    hb = call("/api/messages/conversations", tb)[1]["contacts"][0]["conversation_handle"]
    assert call("/api/messages/images", tb, {"conversation_handle":hb,"enabled":True})[1]["available"]
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=20)
    conn.request("GET", url, headers={"Authorization":"Bearer "+tb,"X-Forwarded-For":"192.0.2.44","X-Forwarded-Proto":"https"})
    response = conn.getresponse()
    assert response.status == 200 and response.read() == png()
    assert response.getheader("Content-Type") == "image/png"
    assert "no-store" in response.getheader("Cache-Control")
    assert response.getheader("X-Content-Type-Options") == "nosniff"
    conn.close()
    assert call(url, tc)[0] == 403
    assert call("/api/messages/thread?peer=ann", tb)[1]["messages"][0]["id"] == receipt["id"]
    assert call("/api/messages/images", tb, {"conversation_handle":hb,"enabled":False})[0] == 200
    assert call(url, ta)[0] == 403
