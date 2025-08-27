# V.E.N.T.R.I.S. - Visual & Executable Navigational Tool for Repetitive & Instantaneous Shortcuts

V.E.N.T.R.I.S. is a Python-based application that allows you to launch applications and shortcuts using American Sign Language (ASL) gestures captured through your webcam. It uses a machine learning model to recognize gestures and maps them to specific actions on your computer.

## Features

- **Gesture Recognition**: Recognizes ASL alphabet gestures in real-time.
- **Customizable Mappings**: Easily configure which gesture launches which application.
- **Automatic Shortcut Discovery**: Scans your system for available shortcuts to map.
- **User-Friendly Interface**: A simple GUI for mapping gestures to your favorite applications.

## How It Works

The application operates in three main stages:

1.  **Shortcut Discovery (`search.py`)**: Automatically searches for `.lnk` files in common directories (Desktop, Start Menu) and copies them into a local directory for easy access.
2.  **Configuration (`config_gui.py`)**: Provides a graphical interface where you can see each ASL gesture image and assign a discovered shortcut to it from a dropdown menu.
3.  **Gesture Recognition & Launch (`ventris_gui.py`)**: Activates your webcam, recognizes your hand gestures, and launches the application you've mapped to that gesture.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/sayan23bcy6/V.E.N.T.R.I.S.git
    cd V.E.N.T.R.I.S
    ```

2.  **Install the required packages:**
    Make sure you have Python 3 installed. Then, run the following command in the project directory to install all necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the application, simply run the `main.py` script:

```bash
python main.py
```

This will initiate the following sequence:

1.  The script will first check if all required packages are installed.
2.  It will then run the shortcut discovery process.
3.  After that, the configuration window will appear. Here, you can map ASL gestures to the shortcuts found on your system. Click "Save Mappings" when you are done.
4.  Finally, the main gesture recognition window will open, your webcam will be activated, and you can start using gestures to launch your applications.

## Files

- `main.py`: The entry point of the application.
- `search.py`: Discovers and collects shortcuts.
- `config_gui.py`: The GUI for mapping gestures to shortcuts.
- `ventris_gui.py`: The main application for gesture recognition.
- `ventris_model_final.pkl`: The pre-trained machine learning model for gesture recognition.
- `requirements.txt`: A list of all Python dependencies.
- `asl_alphabet_test/`: A directory containing images of ASL gestures used in the configuration GUI.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
