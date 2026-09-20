"""Upload magic-byte validation."""
from io import BytesIO

from app.services.resume_parser import UploadValidationError, validate_upload


class _FakeStorage:
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self._data = data

    def read(self):
        return self._data

    def seek(self, *_args, **_kwargs):
        return 0


def test_reject_non_pdf_masquerading():
    fake = _FakeStorage("evil.pdf", b"not a pdf")
    try:
        validate_upload(fake)
        assert False, "should raise"
    except UploadValidationError as exc:
        assert "PDF" in str(exc) or "魔数" in str(exc)


def test_accept_utf8_text():
    fake = _FakeStorage("cv.txt", "姓名：测试\n技能：Python".encode("utf-8"))
    name, data = validate_upload(fake)
    assert name.endswith(".txt")
    assert b"Python" in data


def test_accept_docx_magic():
    # Minimal ZIP that looks like docx (PK header) — content check needs document.xml
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document></w:document>")
        zf.writestr("[Content_Types].xml", "<Types></Types>")
    fake = _FakeStorage("cv.docx", buf.getvalue())
    name, data = validate_upload(fake)
    assert name.endswith(".docx")
    assert data.startswith(b"PK")


def test_reject_docx_without_document_xml():
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "nope")
    fake = _FakeStorage("cv.docx", buf.getvalue())
    try:
        validate_upload(fake)
        assert False, "should raise"
    except UploadValidationError:
        pass
