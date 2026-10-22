"""Local model catalogue using the pinned llama-server web UI's model API shape."""
from pathlib import Path
from .config import path


class Models:
    def __init__(self, cfg):
        raw = cfg.get('models') or []
        if isinstance(raw, str):
            raw = [s.strip() for s in raw.split(',') if s.strip()]
        if not isinstance(raw, list):
            raise ValueError('models must be a list or comma-separated paths')
        self.rows = {}
        for item in raw:
            if not isinstance(item, (str, dict)):
                raise ValueError('Each model must be a path or object')
            row = {'path': item, 'id': Path(item).stem} if isinstance(item, str) else dict(item)
            if set(row) - {'id', 'path', 'lora'} or not row.get('path'):
                raise ValueError('Each model needs path and optional id/lora')
            row.setdefault('id', Path(row['path']).stem)
            self.add(row)
        configured = str(path(cfg['model'])) if cfg.get('model') else ''
        matching = [name for name, row in self.rows.items() if row['path'] == configured]
        if not self.rows or configured and not matching:
            self.add({'id': cfg['model_name'], 'path': cfg.get('model', ''), 'lora': cfg.get('lora', [])})
            matching = [cfg['model_name']]
        self.default = matching[0] if matching else next(iter(self.rows))
        if 'lora' not in self.rows[self.default]:
            self.rows[self.default]['lora'] = list(cfg.get('lora', []))
        resident = cfg.get('resident') or []
        if isinstance(resident, str):
            resident = [s.strip() for s in resident.split(',') if s.strip()]
        if not isinstance(resident, list) or len(resident) > 2:
            raise ValueError('resident admits at most two registered model names/paths')
        self.resident = set()
        for name in resident:
            matches = [key for key, row in self.rows.items() if key == name or row['path'] == str(path(name))]
            if len(matches) != 1:
                raise ValueError('resident must uniquely identify a registered model')
            self.resident.add(matches[0])

    def add(self, row):
        name = row.get('id')
        if not isinstance(name, str) or not name.strip() or len(name) > 200 or any(ord(c) < 32 for c in name):
            raise ValueError('Model id must be a nonempty name of at most 200 characters')
        if name in self.rows:
            raise ValueError('Duplicate model id: ' + name)
        if not isinstance(row['path'], str):
            raise ValueError('Model path must be a string')
        if 'lora' in row and (not isinstance(row['lora'], list) or not all(isinstance(x, str) for x in row['lora'])):
            raise ValueError('Model lora must be a list of adapter paths')
        self.rows[name] = {**row, 'path': str(path(row['path'])) if row['path'] else ''}

    def resolve(self, name=None):
        name = self.default if name is None else name
        if not isinstance(name, str) or name not in self.rows:
            raise ValueError('Unknown model; select an id from /v1/models')
        return self.rows[name]

    def catalog(self, active=None, loading=None):
        return {'object': 'list', 'data': [
            {'id': name, 'object': 'model', 'created': 0, 'owned_by': 'local',
             'status': {'value': 'loading' if name == loading else 'loaded' if name == active else 'unloaded', 'args': []},
             'slstack': {'resident_requested': name in self.resident, 'dual_resident_active': False,
                         'switch_mode': 'unload-load'}} for name in self.rows]}
