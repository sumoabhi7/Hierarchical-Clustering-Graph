# tool_rotate_manual.py
import cv2
import os
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from visualizer import select_query_image_gui
from preprocessing import rotate_image_lossless

def manual_rotation_tool():
    # 1. Select Image
    print("Select an image to rotate...")
    path = select_query_image_gui()
    
    if not path:
        print("No file selected.")
        return

    # Create a hidden root window
    root = tk.Tk()
    root.withdraw() 
    
    # 2. Ask for Angle
    angle_str = simpledialog.askstring("Input", "Enter rotation angle (degrees):\n(e.g. 45, 90, 180)")
    
    if not angle_str:
        print("Operation cancelled.")
        root.destroy()
        return

    try:
        angle = float(angle_str)
    except ValueError:
        messagebox.showerror("Error", "Invalid angle entered.")
        root.destroy()
        return

    # 3. Ask for Destination Folder
    print("Select destination folder...")
    dest_folder = filedialog.askdirectory(title="Select Destination Folder")
    
    if not dest_folder:
        print("No destination folder selected. Saving to source folder.")
        dest_folder = os.path.dirname(path)
    
    root.destroy()

    # 4. Perform Rotation
    print(f"Rotating {os.path.basename(path)} by {angle} degrees...")
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error reading image.")
        return
        
    rotated = rotate_image_lossless(img, angle)
    
    # 5. Save with DB Name Prefix
    # Extract details
    parent_dir = os.path.basename(os.path.dirname(path)) # e.g., "DB1_B"
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    
    # New format: DB1_B_101_1_rot90.tif
    new_filename = f"{parent_dir}_{name}_rot{int(angle)}{ext}"
    save_path = os.path.join(dest_folder, new_filename)
    
    cv2.imwrite(save_path, rotated)
    print(f"✅ Saved rotated image to: {save_path}")
    
    # Optional: Preview
    cv2.imshow("Original", img)
    cv2.imshow(f"Rotated {angle}", rotated)
    print("Press any key on the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    manual_rotation_tool()