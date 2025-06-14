import csv
import os

# Function to create dummy labels for our toy dataset sequentially

def create_csv_labels(video_dir="./data/videos", output_csv='./data/labels/labels.csv', num_classes=5):
    # Collect only valid video files
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(('.avi', '.mp4', '.mov'))])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Write CSV with dummy labels
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['video', 'label'])  # Header
        for idx, filename in enumerate(video_files):
            video_name = os.path.splitext(filename)[0]
            label = idx % num_classes  # Assign labels in round-robin fashion
            writer.writerow([video_name, label])

# Run the function
if __name__ == "__main__":
    create_csv_labels(
        video_dir="./data/videos",
        output_csv="./data/labels/labels.csv",
        num_classes=5
    )