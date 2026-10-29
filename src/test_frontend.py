"""Dependency-free structural checks for the static frontend/API contract."""

import unittest
from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class FrontendMarkupParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.file_input = None
        self.headings = []
        self.buttons = []
        self._capture = None
        self._buffer = ""

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if tag == "input" and attributes.get("type") == "file":
            self.file_input = attributes
        if tag in ("h1", "button"):
            self._capture = tag
            self._buffer = ""

    def handle_data(self, data):
        if self._capture:
            self._buffer += data

    def handle_endtag(self, tag):
        if tag == self._capture:
            value = " ".join(self._buffer.split())
            if tag == "h1":
                self.headings.append(value)
            elif tag == "button":
                self.buttons.append(value)
            self._capture = None


class FrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
        cls.javascript = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
        cls.css = (FRONTEND_DIR / "styles.css").read_text(encoding="utf-8")
        cls.parser = FrontendMarkupParser()
        cls.parser.feed(cls.html)

    def test_static_application_assets_are_present_and_mobile_ready(self):
        self.assertIn("name=\"viewport\"", self.html)
        self.assertIn("href=\"styles.css\"", self.html)
        self.assertIn("src=\"app.js\"", self.html)
        self.assertIn("@media (max-width", self.css)
        self.assertIn("EEG-Based Emotion Recognition", self.parser.headings)

    def test_file_picker_and_actions_match_the_user_flow(self):
        self.assertIsNotNone(self.parser.file_input)
        self.assertIn(".mat", self.parser.file_input.get("accept", ""))
        self.assertEqual(self.parser.file_input.get("name"), "file")
        self.assertIn("predict-button", self.parser.ids)
        self.assertIn("reset-button", self.parser.ids)
        self.assertTrue(any("Predict Emotion" in label for label in self.parser.buttons))
        self.assertTrue(any("Reset / Upload Another File" in label for label in self.parser.buttons))

    def test_client_uses_existing_routes_and_multipart_field(self):
        self.assertIn("/health", self.javascript)
        self.assertIn("/predict", self.javascript)
        self.assertIn('formData.append("file", file, file.name)', self.javascript)
        self.assertIn("new FormData()", self.javascript)
        self.assertNotIn("Content-Type", self.javascript)

    def test_prediction_rendering_uses_api_values_and_handles_errors(self):
        for response_field in (
            "payload.prediction",
            "payload.confidence",
            "payload.class_probabilities",
            "payload.models",
            "payload.window_counts",
            "payload.adaptive_fusion",
            "payload.confidence_description",
        ):
            self.assertIn(response_field, self.javascript)
        for safe_error in (
            "Select a .mat EEG recording",
            "not a .mat recording",
            "backend is unavailable",
            "payload.error.message",
            "The backend returned an incomplete prediction response",
        ):
            self.assertIn(safe_error, self.javascript)
        self.assertIn('textContent = payload.prediction', self.javascript)
        self.assertNotIn("innerHTML", self.javascript)


if __name__ == "__main__":
    unittest.main(verbosity=2)
