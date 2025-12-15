import os
from pathlib import Path
import shutil

# Characters included in ASL Alphabet dataset
ASL_CLASSES = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "space","nothing","delete"
]

def sort_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return
    
    print(f"Sorting folder: {folder}")

    # Create class folders
    for cls in ASL_CLASSES:
        (folder / cls).mkdir(exist_ok=True)

    
    for img in folder.glob("*.*"):
        name = img.name.lower()

        matched = False
        for cls in ASL_CLASSES:
            prefix = cls.lower()
            if name.startswith(prefix):
                shutil.move(str(img), str(folder / cls / img.name))
                matched = True
                break
        
        if not matched:
            print(f"WARNING: Could not classify file: {img}")

    print("Done.")


if __name__ == "__main__":
    base = Path("data/processed")

    # Sort train if exists
    if (base / "asl_alphabet_train").exists():
        sort_folder(base / "asl_alphabet_train")

    if (base / "asl_alphabet_test").exists():
        sort_folder(base / "asl_alphabet_test")

    print("Dataset sorted successfully!")
