import httpx
import pymupdf

from retrieval_lab.corpus import extract_pages, fetch, load_jsonl, make_chunks

FAKE_PDF = b"%PDF-1.4 fake"


def test_chunks_never_cross_pages_and_carry_their_page(pages):
    pages = [{"paper": "p", "page": 3, "text": "One sentence here. " * 60}, {"paper": "p", "page": 4, "text": "Short."}]
    chunks = make_chunks(pages, size=300, overlap=50)
    assert {c.page for c in chunks} == {3, 4}
    assert [c for c in chunks if c.page == 4][0].text == "Short."
    assert all(len(c.text) <= 300 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[0].id == "p:3:0" and chunks[0].page_id == "p#3"


def test_fetch_downloads_politely_skips_what_it_has_and_counts_failures(tmp_path):
    seen, slept = [], []

    def handler(request):
        pid = request.url.path.rsplit("/", 1)[-1]
        seen.append(pid)
        if pid == "bad":
            return httpx.Response(404)
        if pid == "html":
            return httpx.Response(200, content=b"<html>captcha</html>")
        return httpx.Response(200, content=FAKE_PDF)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    (tmp_path / "have.pdf").write_bytes(FAKE_PDF)
    counts = fetch(["have", "a", "bad", "html", "b"], tmp_path, client=client, sleep=slept.append, delay=3.0)
    assert counts == {"have": 1, "copied": 0, "downloaded": 2, "failed": 2}
    assert seen == ["a", "bad", "html", "b"]
    assert slept == [3.0, 3.0, 3.0]  # a pause before every request after the first
    assert not (tmp_path / "html.pdf").exists() and not list(tmp_path.glob("*.part"))

    # resumable: a second pass only retries the failures
    seen.clear()
    fetch(["have", "a", "bad", "html", "b"], tmp_path, client=client, sleep=slept.append)
    assert seen == ["bad", "html"]


def test_fetch_copies_from_a_local_folder_first(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "x.pdf").write_bytes(FAKE_PDF)
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    counts = fetch(["x"], tmp_path / "dest", from_dir=src, client=client, sleep=lambda s: None)
    assert counts["copied"] == 1 and (tmp_path / "dest" / "x.pdf").read_bytes() == FAKE_PDF


def test_extract_pages_reads_real_pdfs_page_by_page(tmp_path):
    doc = pymupdf.open()
    for text in ("first page about retrieval", "", "third page about reranking"):
        doc.new_page().insert_text((72, 72), text)
    doc.save(tmp_path / "p1.pdf")
    out = tmp_path / "pages.jsonl"
    assert extract_pages(["p1", "missing"], tmp_path, out) == 2  # the blank page is dropped
    rows = load_jsonl(out)
    assert [(r["paper"], r["page"]) for r in rows] == [("p1", 1), ("p1", 3)]
    assert "reranking" in rows[1]["text"]


def test_jsonl_survives_unicode_line_separators_inside_text(tmp_path):
    import json

    rows = [{"text": "before\u2028after\x85end\x1cmore"}, {"text": "second"}]
    path = tmp_path / "x.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert load_jsonl(path) == rows
