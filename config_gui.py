
import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QScrollArea, QGridLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

class ConfigGUI(QWidget):
    def __init__(self):
        super().__init__()
        
        # Use AppData\Local for the Ventris directory
        app_data_dir = os.getenv('LOCALAPPDATA')
        if not app_data_dir:
            print("Error: Could not find LOCALAPPDATA environment variable.")
            sys.exit(1)
            
        self.ventris_dir = os.path.join(app_data_dir, "Ventris")
        self.shortcuts_dir = os.path.join(self.ventris_dir, "CollectedShortcuts")
        self.mappings_file = os.path.join(self.ventris_dir, "mappings.json")
        self.asl_test_dir = os.path.join(self.ventris_dir, "asl_alphabet_test")
        
        self.mappings = self.load_mappings()
        self.shortcuts = self.load_shortcuts()
        
        self.initUI()

    def load_mappings(self):
        if os.path.exists(self.mappings_file):
            with open(self.mappings_file, 'r') as f:
                return json.load(f)
        return {}

    def load_shortcuts(self):
        if os.path.exists(self.shortcuts_dir):
            return [f for f in os.listdir(self.shortcuts_dir) if f.endswith('.lnk')]
        return []

    def initUI(self):
        self.setWindowTitle('V.E.N.T.R.I.S. - Configuration')
        self.setGeometry(100, 100, 800, 600)

        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel('Configure Gesture Mappings')
        title_label.setFont(QFont('Arial', 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Scroll Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        scroll_content = QWidget()
        self.grid_layout = QGridLayout(scroll_content)
        self.grid_layout.setSpacing(20)

        self.populate_grid()

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # Save Button
        save_button = QPushButton('Save Mappings')
        save_button.setFont(QFont('Arial', 14))
        save_button.clicked.connect(self.save_mappings)
        main_layout.addWidget(save_button)

        self.setLayout(main_layout)

    def populate_grid(self):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letters = list(alphabet) + ["space"]
        
        row, col = 0, 0
        for letter in letters:
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            frame_layout = QVBoxLayout()

            # Image
            img_label = QLabel()
            img_path = os.path.join(self.asl_test_dir, f"{letter}_test.jpg")
            if os.path.exists(img_path):
                pixmap = QPixmap(img_path)
                img_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                img_label.setText("No Image")
            img_label.setAlignment(Qt.AlignCenter)
            frame_layout.addWidget(img_label)

            # Letter
            letter_label = QLabel(f"Gesture: {letter}")
            letter_label.setFont(QFont('Arial', 12, QFont.Bold))
            letter_label.setAlignment(Qt.AlignCenter)
            frame_layout.addWidget(letter_label)

            # Dropdown
            combo = QComboBox()
            combo.addItem("None")
            combo.addItems(self.shortcuts)
            
            if letter in self.mappings and self.mappings[letter]:
                shortcut_file = os.path.basename(self.mappings[letter])
                if shortcut_file in self.shortcuts:
                    combo.setCurrentText(shortcut_file)

            combo.setObjectName(f"combo_{letter}")
            frame_layout.addWidget(combo)
            
            frame.setLayout(frame_layout)
            self.grid_layout.addWidget(frame, row, col)
            
            col += 1
            if col > 3:
                col = 0
                row += 1

    def save_mappings(self):
        for i in range(self.grid_layout.count()):
            widget = self.grid_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                combo = widget.findChild(QComboBox)
                if combo:
                    letter = combo.objectName().split('_')[1]
                    selected_shortcut = combo.currentText()
                    if selected_shortcut == "None":
                        self.mappings[letter] = None
                    else:
                        self.mappings[letter] = os.path.join(self.shortcuts_dir, selected_shortcut)
        
        with open(self.mappings_file, 'w') as f:
            json.dump(self.mappings, f, indent=4)
            
        print("Mappings saved successfully!")
        self.close()

if __name__ == '__main__':
    app_data_dir = os.getenv('LOCALAPPDATA')
    if not app_data_dir:
        print("Error: Could not find LOCALAPPDATA environment variable.")
        sys.exit(1)
    ventris_dir = os.path.join(app_data_dir, "Ventris")
    
    if not os.path.exists(ventris_dir):
        print("Ventris directory not found. Please run search.py first.")
        sys.exit()
        
    app = QApplication(sys.argv)
    ex = ConfigGUI()
    ex.show()
    sys.exit(app.exec_())
