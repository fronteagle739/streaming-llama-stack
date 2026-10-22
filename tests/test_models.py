import unittest
from unittest.mock import Mock, patch
from slstack.models import Models
from slstack.engines import Engines


class ModelSelectionTests(unittest.TestCase):
    def config(self):
        return {'model': 'first.gguf', 'model_name': 'first', 'models': [
            {'id': 'first', 'path': 'first.gguf', 'lora': []},
            {'id': 'second', 'path': 'second.gguf', 'lora': ['adapter']}],
            'resident': ['first', 'second'], 'lora': [], 'runtime_root': 'runtime'}

    def test_catalog_uses_router_ui_shape_and_does_not_publish_paths(self):
        models = Models(self.config())
        rows = models.catalog('first')['data']
        self.assertEqual([r['id'] for r in rows], ['first', 'second'])
        self.assertEqual([r['status']['value'] for r in rows], ['loaded', 'unloaded'])
        self.assertTrue(all(r['slstack']['resident_requested'] for r in rows))
        self.assertFalse(any(r['slstack']['dual_resident_active'] for r in rows))
        self.assertFalse(any('path' in r for r in rows))

    @patch('slstack.journal.Journal')
    def test_unknown_request_cannot_eject_current_model(self, journal):
        engine = Engines(self.config()); engine.close = Mock()
        with self.assertRaisesRegex(ValueError, 'Unknown model'):
            engine.select('not-registered')
        engine.close.assert_not_called()

    @patch('slstack.journal.Journal')
    def test_switch_reuses_close_and_selects_adapters_per_model(self, journal):
        engine = Engines(self.config()); engine.close = Mock()
        engine.select('second')
        engine.close.assert_called_once()
        self.assertEqual(engine.cfg['model_name'], 'second')
        self.assertEqual(engine.cfg['lora'], ['adapter'])
        engine.select('second')
        engine.close.assert_called_once()
        engine.select('first')
        self.assertEqual(engine.cfg['lora'], [])
        self.assertEqual(engine.close.call_count, 2)

    def test_duplicate_and_unregistered_resident_names_refused(self):
        cfg = self.config(); cfg['models'].append({'id': 'first', 'path': 'third.gguf'})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            Models(cfg)
        cfg = self.config(); cfg['resident'] = ['missing']
        with self.assertRaisesRegex(ValueError, 'registered'):
            Models(cfg)

    @patch('slstack.journal.Journal')
    def test_failed_job_construction_cannot_leave_a_live_attachment_claim(self, journal):
        engine = Engines(self.config())
        engine.attach_entry = 'owned-entry'
        engine._ready = Mock(side_effect=MemoryError('admission refused'))
        with self.assertRaisesRegex(MemoryError, 'admission refused'):
            engine.ready()
        journal.return_value.update.assert_called_once_with('owned-entry', 'failed', applied=False,
            error='Engine constructor failed before creating an owned job')
        self.assertIsNone(engine.attach_entry)
