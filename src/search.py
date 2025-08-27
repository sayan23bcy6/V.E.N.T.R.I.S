import os
import shutil
import json
import sys

def main():
    # Use AppData\Local for the Ventris directory to avoid needing admin rights
    app_data_dir = os.getenv('LOCALAPPDATA')
    if not app_data_dir:
        print("Error: Could not find LOCALAPPDATA environment variable.")
        os.system("pause")
        return

    ventris_dir = os.path.join(app_data_dir, "Ventris")
    shortcuts_dir = os.path.join(ventris_dir, "CollectedShortcuts")
    mappings_file = os.path.join(ventris_dir, "mappings.json")
    asl_test_dir_src = "asl_alphabet_test"
    asl_test_dir_dest = os.path.join(ventris_dir, "asl_alphabet_test")

    print(f"Creating directory: {ventris_dir}")
    os.makedirs(ventris_dir, exist_ok=True)
    
    print(f"Creating directory: {shortcuts_dir}")
    os.makedirs(shortcuts_dir, exist_ok=True)

    if os.path.exists(asl_test_dir_src):
        print(f"Copying '{asl_test_dir_src}' to '{asl_test_dir_dest}'")
        if os.path.exists(asl_test_dir_dest):
            shutil.rmtree(asl_test_dir_dest)
        shutil.copytree(asl_test_dir_src, asl_test_dir_dest)
    else:
        print(f"Warning: '{asl_test_dir_src}' not found. The config GUI may not show images.")

    print("Searching for .lnk files. This may take a moment...")
    
    # Search common user-specific and public locations for shortcuts
    search_paths = [
        os.path.expanduser("~\\Desktop"),
        os.path.expanduser("~\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs"),
        "C:\\ProgramData\\Microsoft\\Windows\\Start Menu",
        os.path.join(os.environ.get("PUBLIC", "C:\\Users\\Public"), "Desktop"),
        os.path.expanduser("~\\AppData\\Roaming\\Microsoft\\Internet Explorer\\Quick Launch")
    ]

    for path in search_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(".lnk"):
                        source_path = os.path.join(root, file)
                        destination_path = os.path.join(shortcuts_dir, file)
                        if not os.path.exists(destination_path):
                            try:
                                shutil.copy2(source_path, destination_path)
                                print(f"Copied: {source_path}")
                            except (IOError, OSError) as e:
                                print(f"Could not copy {source_path}: {e}")

    if not os.path.exists(mappings_file):
        print(f"Creating mappings file: {mappings_file}")
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        mappings = {letter: None for letter in alphabet}
        mappings["space"] = None
        with open(mappings_file, 'w') as f:
            json.dump(mappings, f, indent=4)

    print("\nSetup complete.")
    print(f"Shortcuts collected in: {shortcuts_dir}")
    print(f"Mappings file created at: {mappings_file}")
    print("You can now run config_gui.py to set up your gesture mappings.")
    os.system("pause")


if __name__ == "__main__":
    main()
