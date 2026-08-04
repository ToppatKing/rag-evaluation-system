"""Tests for rag_eval.ingestion.preprocessor."""

from __future__ import annotations

from rag_eval.ingestion.preprocessor import clean_text, remove_boilerplate


class TestPreprocessor:
    def test_remove_boilerplate_preserves_contract_headings(self) -> None:
        text = """Confidentiality
This Agreement is made between the parties.
"""
        cleaned = remove_boilerplate(text)
        assert "Confidentiality" in cleaned
        assert "This Agreement is made between the parties." in cleaned

    def test_remove_boilerplate_strips_page_numbers_and_footer(self) -> None:
        text = """This is important legal text.
Page 2
All rights reserved.
"""
        cleaned = remove_boilerplate(text)
        assert "Page 2" not in cleaned
        assert "All rights reserved" not in cleaned
        assert "This is important legal text." in cleaned

    def test_clean_text_falls_back_when_boilerplate_removes_all_content(self) -> None:
        text = """Page 3
All rights reserved.
"""
        cleaned = clean_text(text)
        assert cleaned.strip() != ""
        assert "Page 3" in cleaned
        assert "All rights reserved" in cleaned
