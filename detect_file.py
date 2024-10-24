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

if __name__ == "__main__":
    img_path = 'Images/Image1.jpg'
    output_dir = 'output/'

    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    output_filename = f'{base_filename}_with_detections.jpg'

    result = detect(img_path)

    # Plot result
    plot_result(result)

    # Print result
    print_result(result)

    # Save to folder
    save_result(result, output_dir, output_filename)


    
