from unittest import TestCase
from fastapi.testclient import TestClient
from image_segmentation.app.application import create_application


class TestBaseEventHandler(TestCase):
    def test_startup_handler(self):
        app = create_application()
        with self.assertLogs('image_segmentation', level='INFO') as cm:

            with TestClient(app):
                pass
            self.assertEqual(cm.output,
                             ['INFO:image_segmentation:Starting up ...',
                              'INFO:image_segmentation:Shutting down ...'])
