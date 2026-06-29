from PySide6.QtWidgets import QApplication
import sys
from app.main_window_v3 import PANCAM_UI


# ============================================================
# APP ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PANCAM_UI()
    window.show()
    sys.exit(app.exec())