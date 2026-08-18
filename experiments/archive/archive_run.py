import os
import shutil
import json
import argparse
from datetime import datetime

def archive(tag):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{timestamp}_{tag}"
    archive_dir = os.path.join("experiments", "history", archive_name)
    
    print(f"Creating archive directory: {archive_dir}")
    os.makedirs(archive_dir, exist_ok=True)
    
    # 1. Copy results/
    results_src = "results"
    if os.path.exists(results_src):
        shutil.copytree(results_src, os.path.join(archive_dir, "results"), dirs_exist_ok=True)
        print("  Copied results/ directory.")
        
    # 2. Copy data/dataset/
    dataset_src = os.path.join("data", "dataset")
    if os.path.exists(dataset_src):
        shutil.copytree(dataset_src, os.path.join(archive_dir, "dataset"), dirs_exist_ok=True)
        print("  Copied data/dataset/ directory.")
        
    # 3. Copy index.html and Report/V1.HTML
    for html_file in ["index.html", os.path.join("Report", "V1.HTML")]:
        if os.path.exists(html_file):
            dest_file = os.path.join(archive_dir, os.path.basename(html_file) if "Report" not in html_file else "Report_V1.HTML")
            shutil.copy2(html_file, dest_file)
            print(f"  Copied {html_file}.")
            
    # 4. Save metadata.json
    metadata = {
        "timestamp": timestamp,
        "tag": tag,
        "description": f"Experimental run archived with tag: {tag}"
    }
    with open(os.path.join(archive_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        print("  Saved metadata.json.")
        
    print(f"\nSuccessfully archived run as '{archive_name}'!")
    print(f"Archive path: {os.path.abspath(archive_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archive experimental dataset and results.")
    parser.add_argument("tag", type=str, help="Tag/Name for this experimental run (e.g. v3_bias, v4_leak_free)")
    args = parser.parse_args()
    
    archive(args.tag)
