import subprocess
import sys
import os
import importlib.util

# List of required packages
REQUIRED_PACKAGES = [
    "PyQt5", "opencv-python", "numpy", "mediapipe", "joblib", "scikit-learn"
]

def install_package(package):
    """Installs a package using pip."""
    print(f"Installing {package}...")
    # Use --user to avoid needing admin rights
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--user"])

def check_and_install_packages():
    """Checks if required packages are installed and installs them if not."""
    print("--- Checking for required packages ---")
    for package in REQUIRED_PACKAGES:
        # Handle packages with different import names
        import_name = package
        if package == "opencv-python":
            import_name = "cv2"
        elif package == "scikit-learn":
            import_name = "sklearn"
        
        # Check if the package is importable
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print(f"Package '{package}' not found. Installing...")
            try:
                install_package(package)
            except subprocess.CalledProcessError as e:
                print(f"--- ERROR: Failed to install {package}. Please install it manually. Error: {e} ---")
                os.system("pause")
                sys.exit(1)
        else:
            print(f"Package '{package}' is already installed.")
    print("--- All required packages are present. ---\n")


def run_script(script_name):
    """Runs a Python script and waits for it to complete."""
    print(f"--- Running {script_name} ---")
    try:
        # Using subprocess.run to wait for the process to complete
        result = subprocess.run([sys.executable, script_name], check=False) # check=False to handle non-zero exits manually
        if result.returncode != 0:
            print(f"--- WARNING: {script_name} exited with code {result.returncode}. ---")
            # For config_gui, a non-zero exit might just mean the user closed it.
            # We can decide if this is a fatal error on a per-script basis.
        else:
            print(f"--- {script_name} finished successfully. ---")
        return True
    except FileNotFoundError:
        print(f"--- ERROR: Script '{script_name}' not found. Make sure it's in the same directory. ---")
        return False
    except Exception as e:
        print(f"--- ERROR: An unexpected error occurred while running {script_name}: {e} ---")
        return False

def main():
    """Main execution flow for V.E.N.T.R.I.S."""
    print("--- V.E.N.T.R.I.S. Launcher Initializing ---")
    
    # Step 0: Ensure all required packages are installed
    check_and_install_packages()

    # Step 1: Run the search script to find all shortcuts
    print("Step 1: Searching for application shortcuts...")
    if not run_script("src/search.py"):
        print("Could not complete the shortcut search. Exiting.")
        os.system("pause")
        sys.exit(1)

    # Step 2: Run the configuration GUI for the user to map gestures
    print("\nStep 2: Please configure your gesture-to-application mappings in the window.")
    print("Save your mappings and close the configuration window to proceed.")
    if not run_script("src/config_gui.py"):
        # This is not necessarily a fatal error, the user might just close it.
        # The main app will handle the case where mappings are not set.
        print("Configuration window closed.")
        
    # Step 3: Launch the main gesture recognition application
    print("\nStep 3: Starting the main gesture recognition application...")
    if not run_script("src/ventris_gui.py"):
        print("The main application has closed or encountered an error.")

    print("\n--- V.E.N.T.R.I.S. has been shut down. ---")
    os.system("pause")

if __name__ == "__main__":
    main()
