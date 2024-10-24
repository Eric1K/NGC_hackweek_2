"""
Run in command line for single image: yolo task=detect mode=predict model=yolo11x.pt source=Images/Image1.jpg
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

def detect(path):
    """
    Runs model on a single file
    """

    # Load the YOLO11 model. YOU MIGHT NEED TO DOWNLOAD SINCE GITHUB FILE SIZE CAP IS 100MB
    model = YOLO('yolo11x.pt')

    # Read image
    
    img = cv2.imread(path)

    # Run model on image
    return model(img)

def plot_result(result):
    """
    Plots the result using matplotlib
    """

    result[0].plot()
    plt.imshow(result[0].plot(show=False)) # show=False hides temp image
    plt.axis('off') 
    plt.show()

def print_result(result):
    print("Bounding boxes:", result[0].boxes.xyxy)  # Coordinates of boxes
    print("Labels:", result[0].boxes.cls)  # Predicted class labels
    print("Confidence scores:", result[0].boxes.conf)  # Confidence scores

def save_result(result, output_dir, filename):
    """
    Saves the result with detections to output folder
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output path
    output_path = os.path.join(output_dir, filename)
    
    # Plot the result 
    result_img = result[0].plot(show=False) 
    plt.imshow(result_img)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close() 
    
    print(f"Image saved to {output_path}")

def process_folder(input_folder, output_folder):
    """
    Processes all images in the input folder, and saves to output folder
    """
    
    # List all files in the input folder
    files = os.listdir(input_folder)

    # Process each file in the folder
    for file in files:
        img_path = os.path.join(input_folder, file)

        # Detect different files
        if file.endswith(('.jpg', '.jpeg', '.png')):
            print(f"Processing {file}...")

            result = detect(img_path)

            print_result(result)

            save_result(result, output_folder, file)

if __name__ == "__main__":
    input_folder = 'prenotedimage/'  # Folder containing images
    output_folder = 'prenoteoutput/'  # Folder where output images will be saved
    
    # Process all images in the input folder and saves to output folder
    process_folder(input_folder, output_folder)


    
