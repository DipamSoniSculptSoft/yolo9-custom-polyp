import streamlit as st
import os
import glob
import pandas as pd
from PIL import Image
import numpy as np
from src.constants import PROJECT_ROOT, DATASET_DIR

# Set page configuration
st.set_page_config(page_title="Image Comparison", layout="wide")

@st.cache_data
def load_images(ORIGINAL_IMAGES_DIR, PREDICTED_IMAGES_DIR):
    # Get list of image files from both directories
    original_images = glob.glob(os.path.join(ORIGINAL_IMAGES_DIR, '*.png')) + \
                     glob.glob(os.path.join(ORIGINAL_IMAGES_DIR, '*.jpg')) + \
                     glob.glob(os.path.join(ORIGINAL_IMAGES_DIR, '*.jpeg'))
    
    predicted_images = glob.glob(os.path.join(PREDICTED_IMAGES_DIR, '*.png')) + \
                      glob.glob(os.path.join(PREDICTED_IMAGES_DIR, '*.jpg')) + \
                      glob.glob(os.path.join(PREDICTED_IMAGES_DIR, '*.jpeg'))

    # Create dictionary to store matched images
    image_pairs = []
    
    # Match images by filename (without extension)
    for orig_path in original_images:
        orig_filename = os.path.splitext(os.path.basename(orig_path))[0]
        # Find matching Polyp Detected Image
        for pred_path in predicted_images:
            pred_filename = os.path.splitext(os.path.basename(pred_path))[0]
            if orig_filename == pred_filename:
                image_pairs.append({
                    'original_path': orig_path,
                    'predicted_path': pred_path
                })
                break
    
    return image_pairs

def main():

    option = st.selectbox('Select Prediction', ('Train Data', 'Unseen Data'))
        
    if option == 'Train Data':

        # Define directory paths
        ORIGINAL_IMAGES_DIR = os.path.join(DATASET_DIR, 'polyp', 'images')
        PREDICTED_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'output', 'predictions', 'train_val')

        # Set title for the app
        st.title("Original vs Polyp Detected Images from Training data Comparison")
        
        # Load image pairs
        image_pairs = load_images(ORIGINAL_IMAGES_DIR, PREDICTED_IMAGES_DIR)
        
        if not image_pairs:
            st.error("No matching image pairs found in the specified directories!")
            return
        
        # Create DataFrame for display
        data = []
        for idx, pair in enumerate(image_pairs):
            data.append({
                'Index': idx + 1,
                'Original Image': pair['original_path'],
                'Polyp Detected Image': pair['predicted_path']
            })
        
        df = pd.DataFrame(data)
        
        # Display table
        st.subheader("Image Comparison Table")
        
        # Create columns for table-like display
        col1, col2, col3 = st.columns([1, 3, 3])
        
        # Display headers
        with col1:
            st.markdown("**Index**")
        with col2:
            st.markdown("**Original Image**")
        with col3:
            st.markdown("**Polyp Detected Image**")
        
        # Display each row
        for idx, row in df.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 3])
                
                with col1:
                    st.write(row['Index'])
                
                with col2:
                    try:
                        img = Image.open(row['Original Image'])
                        # Resize image for display while maintaining aspect ratio
                        img.thumbnail((300, 300))
                        st.image(img, use_container_width=False)
                    except Exception as e:
                        st.error(f"Error loading original image: {e}")
                
                with col3:
                    try:
                        img = Image.open(row['Polyp Detected Image'])
                        # Resize image for display while maintaining aspect ratio
                        img.thumbnail((300, 300))
                        st.image(img, use_container_width=False)
                    except Exception as e:
                        st.error(f"Error loading Polyp Detected Image: {e}")
                
                st.markdown("---")  # Separator between rows

    elif option == 'Unseen Data':
        # Define directory paths
        ORIGINAL_IMAGES_DIR = "/home/lf-158/PROJECTS/GastroIQ/hyper-kvasir-segmented-images/segmented-images/images"
        PREDICTED_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'output', 'predictions', 'unseen')

        # Set title for the app
        st.title("Original vs Polyp Detected Images from Unseen data Comparison")

        # Load image pairs
        image_pairs = load_images(ORIGINAL_IMAGES_DIR, PREDICTED_IMAGES_DIR)
        
        if not image_pairs:
            st.error("No matching image pairs found in the specified directories!")
            return
        
        # Create DataFrame for display
        data = []
        for idx, pair in enumerate(image_pairs):
            data.append({
                'Index': idx + 1,
                'Original Image': pair['original_path'],
                'Unseen Polyp Detected Image': pair['predicted_path']
            })
        
        df = pd.DataFrame(data)
        
        # Display table
        st.subheader("Image Comparison Table")
        
        # Create columns for table-like display
        col1, col2, col3 = st.columns([1, 3, 3])
        
        # Display headers
        with col1:
            st.markdown("**Index**")
        with col2:
            st.markdown("**Original Image**")
        with col3:
            st.markdown("**Unseen Polyp Detected Image**")
        
        # Display each row
        for idx, row in df.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 3])
                
                with col1:
                    st.write(row['Index'])
                
                with col2:
                    try:
                        img = Image.open(row['Original Image'])
                        # Resize image for display while maintaining aspect ratio
                        img.thumbnail((300, 300))
                        st.image(img, use_container_width=False)
                    except Exception as e:
                        st.error(f"Error loading original image: {e}")
                
                with col3:
                    try:
                        img = Image.open(row['Unseen Polyp Detected Image'])
                        # Resize image for display while maintaining aspect ratio
                        img.thumbnail((300, 300))
                        st.image(img, use_container_width=False)
                    except Exception as e:
                        st.error(f"Error loading Unseen Polyp Detected Image: {e}")
                
                st.markdown("---")  # Separator between rows

if __name__ == "__main__":
    main()