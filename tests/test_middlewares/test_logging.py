import unittest.mock as mock
from unittest import TestCase
from fastapi.testclient import TestClient
from image_segmentation.app.application import create_application


class TestLogTime(TestCase):
    def setUp(self):
        app = create_application()
        self.test_client = TestClient(app)

    @mock.patch("image_segmentation.app.middlewares.logging.time")
    def test_log_time(self, mocked_time):
        mocked_time.time.side_effect = [1, 2, 1, 2, 1, 2]
        with self.assertLogs('image_segmentation', level='INFO') as cm:
            version_response = self.test_client.get("/api/v1/version")
            error_response = self.test_client.get("/api/v1/not_exist_page")
            self.assertEqual(version_response.status_code, 200)
            self.assertEqual(error_response.status_code, 404)
            self.assertEqual(cm.output,
                             ['INFO:image_segmentation:testclient:50000 - '
                              '"GET /api/v1/version HTTP/1.1" '
                              '200 OK - 1000.00ms',
                              'INFO:image_segmentation:testclient:50000 - '
                              '"GET /api/v1/not_exist_page HTTP/1.1'
                              '" 404 Not Found - '
                              '1000.00ms'])
