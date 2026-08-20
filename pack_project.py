import os
import argparse
from pathlib import Path

# Define file extensions to ignore (binary and heavy files)
DEFAULT_IGNORE_EXTENSIONS = {
    # Binaries / Images
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.tar', '.gz', '.7z', 
    '.exe', '.dll', '.so', '.dylib', '.bin', '.mp3', '.mp4', '.mov', '.avi',
    # Lockfiles and system
    '.lock', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', '.DS_Store', 'Thumbs.db',
    # Compiled code
    '.pyc', '.pyo', '.class', '.o', '.obj'
}

# Define folder names to ignore
DEFAULT_IGNORE_FOLDERS = {
    'node_modules', '.git', '.github', '__pycache__', 'dist', 'build', '.next', '.venv', 'venv', 'env'
}

def is_text_file(file_path):
    """Check if a file is plain text."""
    if file_path.suffix.lower() in DEFAULT_IGNORE_EXTENSIONS:
        return False
    try:
        with open(file_path, 'tr', encoding='utf-8') as f:
            f.read(512) # Try reading a small chunk
        return True
    except (UnicodeDecodeError, PermissionError):
        return False

def generate_context(root_dir, output_file):
    root_path = Path(root_dir).resolve()
    
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("==================================================\n")
        out.write(f"PROJECT DIRECTORY TREE FOR: {root_path.name}\n")
        out.write("==================================================\n\n")
        
        # 1. Generate visual directory map
        for root, dirs, files in os.walk(root_path):
            # Modify dirs in-place to skip ignored folders
            dirs[:] = [d for d in dirs if d not in DEFAULT_IGNORE_FOLDERS]
            
            level = len(Path(root).relative_to(root_path).parts)
            indent = '  ' * level
            if level == 0:
                out.write(f".\n")
            else:
                out.write(f"{indent}├── {Path(root).name}/\n")
                
            sub_indent = '  ' * (level + 1)
            for f in files:
                f_path = Path(root) / f
                if f_path.suffix.lower() not in DEFAULT_IGNORE_EXTENSIONS and f not in DEFAULT_IGNORE_EXTENSIONS:
                    out.write(f"{sub_indent}└── {f}\n")
        
        out.write("\n\n==================================================\n")
        out.write("FILE CONTENTS\n")
        out.write("==================================================\n\n")
        
        # 2. Append file contents
        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if d not in DEFAULT_IGNORE_FOLDERS]
            
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(root_path)
                
                if is_text_file(file_path):
                    out.write(f"--- START OF FILE: {relative_path} ---\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            out.write(f.read())
                    except Exception as e:
                        out.write(f"[ERROR READING FILE: {e}]\n")
                    out.write(f"\n--- END OF FILE: {relative_path} ---\n\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pack a directory into a single text file for LLMs.")
    parser.add_argument('-d', '--dir', default='.', help='Directory to pack (default: current directory)')
    parser.add_argument('-o', '--output', default='claude_project_context.txt', help='Output filename')
    args = parser.parse_args()
    
    print(f"Packing {args.dir} into {args.output}...")
    generate_context(args.dir, args.output)
    print("Done! Upload the generated file to your Claude Project.")
