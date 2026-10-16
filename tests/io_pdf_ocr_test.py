"""OCR recovery at the adapter boundary; no models, network, or source books."""
from __future__ import annotations
from tests.local_source_controls import private_value
import hashlib
import json
import os
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
from pdf_chunker.adapters import io_pdf

def _block(text: str, page: int, **source: object) -> dict:
    return {'type': 'paragraph', 'text': text, 'source': {'page': page, **source}}

def _ocr_tuple(text: str, index: int=0, *, y0: float=100, y1: float=130) -> tuple:
    return (50.0, y0, 400.0, y1, text, index, 0)

class OcrRecoveryTest(unittest.TestCase):

    def setUp(self) -> None:
        self.enterContext(patch.dict(os.environ))
        os.environ.pop('PDF_CHUNKER_OCR_LANGUAGE', None)
        self.path = 'library/Original Source.pdf'
        self.primary = self.enterContext(patch.object(io_pdf, '_primary_blocks', return_value=[]))
        self.fallback = self.enterContext(patch.object(io_pdf, '_fallback_blocks', return_value=[]))
        self.enterContext(patch.object(io_pdf, '_page_numbers', return_value=range(1, 6)))
        self.pages = [SimpleNamespace(rect=SimpleNamespace(height=800), get_textpage_ocr=Mock(return_value=object()), get_text=Mock(return_value=[])) for _ in range(5)]
        self.document = MagicMock()
        self.document.__getitem__.side_effect = self.pages.__getitem__
        self.open_pdf = self.enterContext(patch.object(io_pdf.fitz, 'open', create=True))
        self.open_pdf.return_value.__enter__.return_value = self.document

    def enable(self, language: str='eng') -> None:
        os.environ['PDF_CHUNKER_OCR_LANGUAGE'] = language

    def texts(self, result: dict) -> list[str]:
        return [block['text'] for page in result['pages'] for block in page['blocks']]

    def test_absent_env_preserves_exact_payload_and_does_not_open_for_ocr(self) -> None:
        native = _block('Native text.', 2, index=7)
        self.primary.return_value = [native]
        result = io_pdf.read(self.path)
        self.assertEqual(result, {'type': 'page_blocks', 'source_path': self.path, 'pages': [{'page': page, 'blocks': [native] if page == 2 else []} for page in range(1, 6)], 'extraction_quality': {}})
        self.open_pdf.assert_not_called()

    def test_empty_env_disables_ocr(self) -> None:
        for language in ('', '  '):
            with self.subTest(language=language):
                self.enable(language)
                self.assertNotIn('ocr', io_pdf.read(self.path)['extraction_quality'])
        self.open_pdf.assert_not_called()

    def test_recovers_only_missing_pages_in_source_order_with_provenance(self) -> None:
        self.enable('eng+fra')
        native = _block('Native text.', 1, filename='Original Source.pdf', index=4)
        fallback = _block('Fallback text.', 3, method='pdftotext')
        original = deepcopy(native)
        self.primary.return_value = [native]
        self.fallback.return_value = [fallback, _block('Duplicate native.', 1)]
        self.pages[1].get_text.return_value = [_ocr_tuple('Recovered first.\n', 2), _ocr_tuple('Recovered second.\n', 4, y0=160, y1=190)]
        self.pages[3].get_text.return_value = [_ocr_tuple('Recovered fourth.')]
        result = io_pdf.read(self.path, exclude_pages='5')
        self.assertEqual([page['page'] for page in result['pages']], [1, 2, 3, 4])
        self.assertEqual(self.texts(result), ['Native text.', 'Recovered first.', 'Recovered second.', 'Fallback text.', 'Recovered fourth.'])
        self.assertEqual(native, original)
        self.assertIs(result['pages'][0]['blocks'][0], native)
        self.assertIs(result['pages'][2]['blocks'][0], fallback)
        recovered = result['pages'][1]['blocks'][0]
        self.assertEqual(recovered['type'], 'paragraph')
        self.assertEqual(recovered['bbox'], (50.0, 100, 400.0, 130))
        self.assertEqual(recovered['source'], {'filename': 'Original Source.pdf', 'path': self.path, 'page': 2, 'index': 2, 'location': None, 'method': 'pymupdf_ocr'})
        for page_index in (1, 3):
            page = self.pages[page_index]
            page.get_textpage_ocr.assert_called_once_with(language='eng+fra', dpi=200, full=True)
            page.get_text.assert_called_once_with('blocks', textpage=page.get_textpage_ocr.return_value)
        for page_index in (0, 2, 4):
            self.pages[page_index].get_textpage_ocr.assert_not_called()
        self.fallback.assert_called_once_with(self.path, [5])

    def test_text_bearing_cross_page_ranges_are_not_repeated(self) -> None:
        self.enable()
        self.primary.return_value = [_block('Merged native text.', 1, page_range=(1, 3))]
        io_pdf.read(self.path, exclude_pages=[5])
        self.document.__getitem__.assert_called_once_with(3)
        for page in self.pages[:3]:
            page.get_textpage_ocr.assert_not_called()

    def test_empty_and_whitespace_blocks_do_not_count_as_text_coverage(self) -> None:
        self.enable()
        self.primary.return_value = [_block('', 1), _block(' \n\t', 2, page_range=(2, 3))]
        self.pages[0].get_text.return_value = [_ocr_tuple('One.')]
        self.pages[1].get_text.return_value = [_ocr_tuple('Two.')]
        result = io_pdf.read(self.path, exclude_pages=[3, 4, 5])
        for page in self.pages[:2]:
            page.get_textpage_ocr.assert_called_once()
        self.assertEqual([text for text in self.texts(result) if text.strip()], ['One.', 'Two.'])

    def test_all_populated_or_excluded_skips_ocr_open(self) -> None:
        self.enable()
        self.primary.return_value = [_block('x', 1, page_range=(1, 5))]
        result = io_pdf.read(self.path)
        self.open_pdf.assert_not_called()
        self.assertEqual(result['extraction_quality']['ocr']['attempted_pages'], [])

    def test_all_excluded_never_attempts_ocr(self) -> None:
        self.enable()
        result = io_pdf.read(self.path, exclude_pages='1-5')
        self.assertEqual(result['pages'], [])
        self.open_pdf.assert_not_called()

    def test_blank_and_image_descriptions_do_not_become_fake_text(self) -> None:
        self.enable()
        self.pages[0].get_text.return_value = [_ocr_tuple(' \n'), (0, 0, 500, 800, '<image: DeviceRGB>', 1, 1)]
        result = io_pdf.read(self.path, exclude_pages='2-5')
        self.assertEqual(result['pages'], [{'page': 1, 'blocks': []}])
        self.assertEqual(result['extraction_quality']['ocr']['recovered_pages'], [])
        self.assertEqual(result['extraction_quality']['ocr']['recovered_characters'], 0)

    def test_margins_filter_ocr_blocks_with_strict_boundaries(self) -> None:
        self.enable()
        self.pages[0].get_text.return_value = [_ocr_tuple('Header', 0, y0=49, y1=65), _ocr_tuple('Body', 1, y0=50, y1=760), _ocr_tuple('Footer', 2, y0=750, y1=761)]
        result = io_pdf.read(self.path, exclude_pages='2-5', header_margin=50, footer_margin=40)
        self.assertEqual(self.texts(result), ['Body'])
        self.assertEqual(result['extraction_quality']['ocr']['recovered_blocks'], 1)

    def test_margin_only_ocr_is_not_counted_as_recovered(self) -> None:
        self.enable()
        self.pages[0].get_text.return_value = [_ocr_tuple('Header', y0=10, y1=30)]
        result = io_pdf.read(self.path, exclude_pages='2-5', header_margin=50)
        self.assertEqual(self.texts(result), [])
        self.assertEqual(result['extraction_quality']['ocr']['recovered_pages'], [])

    def test_existing_quality_report_is_retained_with_ocr_counts(self) -> None:
        self.enable()
        quality = {'selected_engine': 'pymupdf', 'score': 0.0, 'missing_pages': [1, 2]}

        def primary(*args: object, **kwargs: object) -> list:
            kwargs['quality_report'].update(deepcopy(quality))
            return []
        self.primary.side_effect = primary
        self.pages[0].get_text.return_value = [_ocr_tuple('Real text.')]
        result = io_pdf.read(self.path, exclude_pages='3-5')
        self.assertEqual(result['extraction_quality'], {**quality, 'ocr': {'language': 'eng', 'dpi': 200, 'attempted_pages': [1, 2], 'recovered_pages': [1], 'recovered_blocks': 1, 'recovered_characters': len('Real text.')}})

    def test_ocr_document_open_error_surfaces_with_path_and_first_missing_page(self) -> None:
        self.enable()
        error = RuntimeError('Cannot open PDF for OCR')
        self.open_pdf.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            io_pdf.read(self.path, exclude_pages=[1, 3, 4, 5])
        self.assertIn(self.path, str(raised.exception))
        self.assertIn('page 2', str(raised.exception))
        self.assertIs(raised.exception.__cause__, error)

    def test_ocr_and_text_extraction_errors_include_original_path_and_page(self) -> None:
        self.enable()
        for method in ('get_textpage_ocr', 'get_text'):
            with self.subTest(method=method):
                error = RuntimeError('OCR engine failed')
                getattr(self.pages[1], method).side_effect = error
                with self.assertRaises(RuntimeError) as raised:
                    io_pdf.read(self.path, exclude_pages=[1, 3, 4, 5])
                self.assertIn(self.path, str(raised.exception))
                self.assertIn('page 2', str(raised.exception))
                self.assertIn('OCR engine failed', str(raised.exception))
                self.assertIs(raised.exception.__cause__, error)
                getattr(self.pages[1], method).side_effect = None

class RealOcrIntegrationTest(unittest.TestCase):

    @unittest.skipUnless(hasattr(io_pdf.fitz, 'Page'), 'PyMuPDF OCR unavailable')
    def test_generated_scanned_pdf_and_blank_page(self) -> None:
        fitz = io_pdf.fitz
        if not hasattr(fitz.Page, 'get_textpage_ocr'):
            self.skipTest('PyMuPDF OCR unavailable')
        with fitz.open() as probe:
            page = probe.new_page()
            page.insert_text((72, 120), 'OCR availability', fontsize=24)
            try:
                page.get_textpage_ocr(language='eng', dpi=200, full=True)
            except RuntimeError as exc:
                unavailable = ('tessdata', 'tesseract', 'traineddata', 'ocr support')
                if any((token in str(exc).lower() for token in unavailable)):
                    self.skipTest(f'OCR unavailable: {exc}')
                raise
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory, fitz.open() as original, fitz.open() as scan:
            page = original.new_page()
            page.insert_text((72, 120), 'SCANNED RECOVERY CHECK', fontsize=24)
            pixmap = page.get_pixmap(dpi=200, alpha=False)
            scanned_page = scan.new_page(width=page.rect.width, height=page.rect.height)
            scanned_page.insert_image(scanned_page.rect, stream=pixmap.tobytes('png'))
            scan.new_page()
            path = Path(directory) / 'Original scan.pdf'
            scan.save(path)
            self.assertEqual(scan[0].get_text().strip(), '')
            with patch.dict(os.environ):
                os.environ.pop('PDF_CHUNKER_OCR_LANGUAGE', None)
                before = io_pdf.read(str(path))
                os.environ['PDF_CHUNKER_OCR_LANGUAGE'] = 'eng'
                after = io_pdf.read(str(path))
            self.assertTrue(all((not page['blocks'] for page in before['pages'])))
            text = ' '.join((block['text'] for block in after['pages'][0]['blocks']))
            self.assertIn('SCANNED RECOVERY CHECK', text.upper())
            self.assertEqual(after['pages'][1]['blocks'], [])
            self.assertEqual(after['extraction_quality']['ocr']['attempted_pages'], [1, 2])
            self.assertEqual(after['extraction_quality']['ocr']['recovered_pages'], [1])
            for block in after['pages'][0]['blocks']:
                self.assertEqual(block['source']['filename'], path.name)
                self.assertEqual(block['source']['path'], str(path))
                self.assertEqual(block['source']['page'], 1)
                self.assertTrue(block['bbox'])

    def test_optional_scanned_book_body_sample(self) -> None:
        path = Path(private_value('control_007', ''))
        if not path.is_file():
            self.skipTest('optional local corpus fixture unavailable')
        expected = private_value('control_008', '')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        selected = {5, 67, 132}
        with io_pdf.fitz.open(path) as document:
            self.assertTrue(all((not page.get_text().strip() for page in document)))
            excluded = [page for page in range(1, len(document) + 1) if page not in selected]
        with patch.dict(os.environ, {'PDF_CHUNKER_OCR_LANGUAGE': 'eng'}):
            result = io_pdf.read(str(path), exclude_pages=excluded)
        counts = {page['page']: sum((len(block['text']) for block in page['blocks'])) for page in result['pages']}
        self.assertEqual(set(counts), selected)
        self.assertTrue(all((count > 100 for count in counts.values())), counts)
        self.assertTrue(all((block['source']['filename'] == path.name for page in result['pages'] for block in page['blocks'])))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        print(json.dumps({'source_file': path.name, 'source_sha256': expected, 'sample_chars_by_page': counts, 'cpu_only': True, 'ocr': result['extraction_quality']['ocr']}))
