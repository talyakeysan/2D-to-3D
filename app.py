
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import numpy as np
from werkzeug.utils import secure_filename
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import open3d as o3d


# Initialize Flask app
app = Flask(__name__)


# Set directories for storing uploaded files
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# İzin verilen dosya türleri
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Gerekli fonksiyonlar

#kamera parametreleri
def get_intrinsics(H, W, fov=45.0):   #fov= görüş alanı 
    f = 0.5 * W / np.tan(0.5 * fov * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


# Function to remove the background (beyaz arkaplan)
def remove_background(color_image, threshold=240):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

    # Create a binary mask where white is the background
    _, mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # Invert mask (background becomes 0, object becomes 1)
    mask = cv2.bitwise_not(mask)

    # Apply mask to each channel in the color image
    masked_image = cv2.bitwise_and(color_image, color_image, mask=mask)

    return masked_image, mask


# Create point cloud but exclude background using a mask
def create_point_cloud(depth_image, color_image, camera_intrinsics=None, scale_ratio=100.0, threshold=240):
    height, width = depth_image.shape

    if camera_intrinsics is None:
        camera_intrinsics = get_intrinsics(height, width)

    # Resize color image to match depth image dimensions
    color_image = cv2.resize(color_image, (width, height))

    # Remove the background from the color image
    color_image_no_bg, mask = remove_background(color_image, threshold=threshold)

    # Ensure depth values are positive and scale them
    depth_image = np.maximum(depth_image, 1e-5)
    depth_image = scale_ratio / depth_image

    # Apply mask to depth image (exclude background pixels)
    depth_image[mask == 0] = 0  # Set depth values in the background to zero

    # Calculate point cloud coordinates (x, y, z)
    x, y, z = pixel_to_point(depth_image, camera_intrinsics)

    # Stack the point cloud
    point_image = np.stack([x, y, z], axis=-1)

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()

    # Exclude points with zero depth (background)
    valid_points = (z > 0)

    # Only add valid points to the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(point_image[valid_points].reshape(-1, 3))
    point_cloud.colors = o3d.utility.Vector3dVector(color_image_no_bg[valid_points].reshape(-1, 3) / 255.0)

    return point_cloud


def pixel_to_point(depth_image, camera_intrinsics=None):
    height, width = depth_image.shape

    if camera_intrinsics is None:
        camera_intrinsics = get_intrinsics(height, width)

    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
#pixel koordinatları
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xx, yy = np.meshgrid(x, y)

    x_over_z = (xx - cx) / fx
    y_over_z = (yy - cy) / fy
# z= derinlik değeri
    z = depth_image / np.sqrt(1. + x_over_z ** 2 + y_over_z ** 2)
    x = x_over_z * z
    y = y_over_z * z

    return x, y, z

""""
def create_point_cloud(depth_image, color_image, camera_intrinsics=None, scale_ratio=100.0):
    height, width = depth_image.shape
    if camera_intrinsics is None:
        camera_intrinsics = get_intrinsics(height, width)

    color_image = cv2.resize(color_image, (width, height))
    depth_image = np.maximum(depth_image, 1e-5)
    depth_image = scale_ratio / depth_image

    x, y, z = pixel_to_point(depth_image, camera_intrinsics)
    point_image = np.stack([x, y, z], axis=-1)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_image.reshape(-1, 3))
    point_cloud.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)
    return point_cloud
"""


# Function to check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function for 3D reconstruction
def reconstruct_3d_from_image(image_path, output_obj_path):

    output_image = cv2.imread(image_path)

    if output_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Load model and processor for depth estimation
    processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf")

    # Process image
    train_input = processor(images=output_image, return_tensors="pt")

    with torch.no_grad():
        train_outputs = model(**train_input)
        train_depth = train_outputs.predicted_depth.squeeze().cpu().numpy()

    # Point cloud oluşturma kısmı
    point_cloud = create_point_cloud(train_depth, output_image)
    output_ply_path = os.path.join(os.path.dirname(output_obj_path), 'output.ply')
    o3d.io.write_point_cloud(output_ply_path, point_cloud)
    print(f"Point cloud saved at {output_ply_path}")

    #estimate normals
    pcd = o3d.io.read_point_cloud(output_ply_path)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

    #kırpılmış 3d mesh
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    p_mesh_crop.compute_triangle_normals()


    o3d.io.write_triangle_mesh(output_obj_path, p_mesh_crop)
    print(f"Mesh saved to {output_obj_path}")
    return output_obj_path

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route for file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in the request.")
        return redirect(request.url)
    
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        print(f"Saving uploaded file to: {file_path}")
        file.save(file_path)

        if os.path.exists(file_path):
            print(f"File successfully saved at: {file_path}")
        else:
            print(f"Error: File could not be saved at {file_path}")

        # Define the output .obj file path
        output_obj_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_model.obj')

        try:
            reconstruct_3d_from_image(file_path, output_obj_path)
            print(f"3D model saved to {output_obj_path}")
        except Exception as e:
            print(f"Error during 3D reconstruction: {e}")
            return "Error during 3D reconstruction", 500

        return send_file(output_obj_path, as_attachment=True)

    print("Invalid file type")
    return "Invalid file type. Only PNG, JPG, and JPEG are allowed.", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
