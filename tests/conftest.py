import gc
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MOLEDITPY_HEADLESS", "1")


def flush_qt(app):
    """Close and really delete every top-level widget.

    Widgets that are still alive when Python tears the QApplication down abort
    the interpreter on Linux ("pure virtual method called"), after the tests
    have already passed.
    """
    from PySide6.QtCore import QCoreApplication, QEvent

    for widget in list(app.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    gc.collect()


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    flush_qt(app)
