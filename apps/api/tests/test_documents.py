"""Tests for document upload flow: presign, upload, confirm."""

import uuid
import pytest

from app.core.config import settings
from app.services.storage import LocalStorage


PRESIGN_BODY = {
    "user_id": "11111111-1111-1111-1111-111111111111",
    "filename": "test.pdf",
    "content_type": "application/pdf",
    "file_size_bytes": 1024,
}


@pytest.fixture
def use_local_storage(monkeypatch, tmp_path):
    """Force LocalStorage with temp dir so we don't touch S3 or project uploads."""
    storage = LocalStorage(base_path=str(tmp_path))

    def _get_storage():
        return storage

    # Patch both so all code paths (presign, confirm, upload-local) use our storage
    monkeypatch.setattr("app.services.storage.get_storage", _get_storage)
    monkeypatch.setattr("app.routers.documents.get_storage", _get_storage)
    return storage


@pytest.mark.asyncio
async def test_presign_success(client, demo_key_off, use_local_storage):
    """Presign returns document_id, s3_key, upload_url, method."""
    resp = await client.post("/documents/presign", json=PRESIGN_BODY)
    assert resp.status_code == 200
    data = resp.json()
    assert "document_id" in data
    assert "s3_key" in data
    assert "upload_url" in data
    assert data["method"] == "PUT"
    assert data["s3_key"].startswith("documents/")
    assert data["s3_key"].endswith("test.pdf") or "test.pdf" in data["s3_key"]


@pytest.mark.asyncio
async def test_presign_rejects_pdf_too_large(client, demo_key_off, use_local_storage, monkeypatch):
    """Presign returns 400 when file exceeds max_pdf_mb."""
    monkeypatch.setattr(settings, "max_pdf_mb", 1)
    body = {**PRESIGN_BODY, "file_size_bytes": 2 * 1024 * 1024}  # 2 MB
    resp = await client.post("/documents/presign", json=body)
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "PDF too large"


@pytest.mark.asyncio
async def test_presign_rejects_invalid_content_type(client, demo_key_off, use_local_storage):
    """Presign returns 422 for non-PDF content type."""
    body = {**PRESIGN_BODY, "content_type": "image/jpeg"}
    resp = await client.post("/documents/presign", json=body)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_presign_rejects_invalid_input(client, demo_key_off, use_local_storage):
    """Presign returns 422 for missing or invalid fields."""
    resp = await client.post("/documents/presign", json={})
    assert resp.status_code == 422

    resp = await client.post("/documents/presign", json={**PRESIGN_BODY, "file_size_bytes": 0})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_full_upload_flow(client, demo_key_off, use_local_storage):
    """Presign -> upload to upload-local -> confirm yields status=uploaded."""
    # Step 1: Presign
    presign_resp = await client.post("/documents/presign", json=PRESIGN_BODY)
    assert presign_resp.status_code == 200
    presign_data = presign_resp.json()
    document_id = presign_data["document_id"]
    s3_key = presign_data["s3_key"]
    upload_url = presign_data["upload_url"]

    # Step 2: Upload (for LocalStorage, upload_url points to our upload-local endpoint)
    pdf_content = b"%PDF-1.4 fake test content"
    from urllib.parse import urlparse
    parsed = urlparse(upload_url)
    put_path = parsed.path + ("?" + parsed.query if parsed.query else "")
    upload_resp = await client.put(put_path, content=pdf_content)
    assert upload_resp.status_code == 200

    # Step 3: Confirm
    confirm_body = {
        "user_id": PRESIGN_BODY["user_id"],
        "document_id": document_id,
        "s3_key": s3_key,
    }
    confirm_resp = await client.post("/documents/confirm", json=confirm_body)
    assert confirm_resp.status_code == 200
    assert confirm_resp.json()["status"] == "uploaded"
    assert confirm_resp.json()["document_id"] == document_id


@pytest.mark.asyncio
async def test_confirm_document_not_found(client, demo_key_off, use_local_storage):
    """Confirm returns 404 when document_id does not exist."""
    fake_id = str(uuid.uuid4())
    confirm_body = {
        "user_id": PRESIGN_BODY["user_id"],
        "document_id": fake_id,
        "s3_key": "documents/fake/fake/test.pdf",
    }
    resp = await client.post("/documents/confirm", json=confirm_body)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_confirm_file_not_in_storage(client, demo_key_off, use_local_storage):
    """Confirm returns 400 when doc exists in DB but file not uploaded yet."""
    presign_resp = await client.post("/documents/presign", json=PRESIGN_BODY)
    assert presign_resp.status_code == 200
    presign_data = presign_resp.json()
    # Don't upload - go straight to confirm
    confirm_body = {
        "user_id": PRESIGN_BODY["user_id"],
        "document_id": presign_data["document_id"],
        "s3_key": presign_data["s3_key"],
    }
    resp = await client.post("/documents/confirm", json=confirm_body)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    detail_str = detail if isinstance(detail, str) else str(detail)
    assert "storage" in detail_str.lower() or "upload" in detail_str.lower()


@pytest.mark.asyncio
async def test_upload_local_creates_file(client, demo_key_off, use_local_storage):
    """PUT upload-local creates the file at the expected path."""
    presign_resp = await client.post("/documents/presign", json=PRESIGN_BODY)
    assert presign_resp.status_code == 200
    s3_key = presign_resp.json()["s3_key"]
    upload_url = presign_resp.json()["upload_url"]

    from urllib.parse import urlparse
    parsed = urlparse(upload_url)
    put_path = parsed.path + ("?" + parsed.query if parsed.query else "")

    pdf_content = b"%PDF-1.4 test"
    resp = await client.put(put_path, content=pdf_content)
    assert resp.status_code == 200

    assert use_local_storage.exists(s3_key)
    from pathlib import Path
    path = Path(use_local_storage.get_path(s3_key))
    assert path.read_bytes() == pdf_content
