"""Graceful abort for combo runs. A supervisor (run_direct) creates the file named by
DS4COMBO_ABORT_FILE; the run then refuses its next native/producer operation at the
boundary (never inside one), so the exception unwinds through the normal close()
paths and every engine exits Studio-style. Cancelling instead would make the
persistent producer terminate its Job, which is the leak pattern."""
import os


class AbortRequested(RuntimeError):
    pass


def requested():
    name = os.environ.get('DS4COMBO_ABORT_FILE')
    return bool(name) and os.path.exists(name)


def check(where):
    if requested():
        raise AbortRequested('DS4COMBO graceful abort requested before ' + where)
