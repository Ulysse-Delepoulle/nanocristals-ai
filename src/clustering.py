"""
Unsupervised clustering of SEM nanocrystal images.

Pipeline:
  1. Crop metadata bar from raw .bmp SEM images and save as .npy
  2. Extract deep features with VGG16 (ImageNet weights, no top)
  3. Reduce dimensionality with PCA
  4. Cluster with KMeans
  5. Sort source images into per-cluster folders for visual inspection

Tested on two zoom levels:
  - Main zoom: ~500 images, 10 clusters
  - Zoom x1:   ~120 images,  6 clusters

Usage:
  python src/clustering.py --bmp-dir path/to/bmp --n-clusters 10
  python src/clustering.py --bmp-dir path/to/bmp_zoom1 --n-clusters 6 --n-components 120
"""

import os
import shutil
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Pixels to crop from raw SEM images to remove the instrument metadata bar.
# Adjust if your images have a different resolution or bar height.
DEFAULT_CROP_ZONE = (0, 0, 1228, 750)


def convert_and_crop_images(img_folder, output_folder, crop_zone=DEFAULT_CROP_ZONE):
    img_names = list(glob.glob(os.path.join(img_folder, "*.bmp")))
    print(f"Preprocessing {len(img_names)} BMP images...")
    os.makedirs(output_folder, exist_ok=True)
    for img_name in img_names:
        image = Image.open(img_name)
        cropped = image.crop(crop_zone).convert("RGB")
        base_name = os.path.basename(img_name).replace(".bmp", ".npy")
        np.save(os.path.join(output_folder, base_name), np.array(cropped))
    print(f"Saved {len(img_names)} arrays to {output_folder}")


def extract_features(image_array, model):
    img = Image.fromarray(image_array).convert("L")
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.concatenate([img, img, img], axis=-1)
    img = np.expand_dims(img, axis=0).astype("float32") / 255.0
    return model.predict(img, verbose=0).flatten()


def run_clustering(numpy_folder, n_components, n_clusters, results_file):
    model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    image_paths = sorted(
        os.path.join(numpy_folder, f)
        for f in os.listdir(numpy_folder)
        if f.endswith(".npy")
    )
    print(f"Extracting VGG16 features from {len(image_paths)} images...")
    features_list = [extract_features(np.load(p), model) for p in image_paths]
    features_array = np.array(features_list)
    print(f"Feature matrix shape: {features_array.shape}")

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features_array)
    print(f"Explained variance (first 2 PCs): {pca.explained_variance_ratio_[:2].sum():.2%}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap="tab10", s=20)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"Nanocrystal Clustering ({n_clusters} clusters)")
    plt.tight_layout()
    plot_path = results_file.replace(".txt", "_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Cluster plot saved to {plot_path}")

    with open(results_file, "w") as f:
        for img_path, cluster in zip(image_paths, clusters):
            f.write(f"{img_path}\tCluster {cluster}\n")
    print(f"Cluster assignments saved to {results_file}")

    return image_paths, clusters


def sort_into_folders(bmp_folder, output_base_folder, results_file):
    with open(results_file) as f:
        lines = f.readlines()

    for line in lines:
        npy_path, cluster = line.strip().split("\t")
        base_name = os.path.basename(npy_path).replace(".BMP.npy", ".BMP")
        cluster_number = cluster.split()[-1]
        bmp_file = os.path.join(bmp_folder, base_name)
        if not os.path.exists(bmp_file):
            print(f"Not found: {bmp_file}")
            continue
        cluster_folder = os.path.join(output_base_folder, f"cluster_{cluster_number}")
        os.makedirs(cluster_folder, exist_ok=True)
        shutil.copy(bmp_file, os.path.join(cluster_folder, base_name))

    print(f"Images sorted into cluster folders under {output_base_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster SEM nanocrystal images using VGG16 + PCA + KMeans"
    )
    parser.add_argument("--bmp-dir", required=True, help="Folder of raw .bmp SEM images")
    parser.add_argument(
        "--numpy-dir", default="numpy_cache", help="Folder to store preprocessed .npy arrays"
    )
    parser.add_argument(
        "--output-dir", default="clusters", help="Root folder for sorted cluster subfolders"
    )
    parser.add_argument("--n-clusters", type=int, default=10, help="Number of KMeans clusters")
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="PCA components (default: min(n_images, 400))",
    )
    parser.add_argument(
        "--crop-zone",
        type=int,
        nargs=4,
        default=list(DEFAULT_CROP_ZONE),
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Crop rectangle to remove SEM metadata bar",
    )
    parser.add_argument(
        "--results-file", default="clustering_results.txt", help="Output file for cluster assignments"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip BMP→numpy step if arrays are already cached",
    )
    args = parser.parse_args()

    if not args.skip_preprocessing:
        convert_and_crop_images(args.bmp_dir, args.numpy_dir, tuple(args.crop_zone))

    n_images = len([f for f in os.listdir(args.numpy_dir) if f.endswith(".npy")])
    n_components = args.n_components or min(n_images, 400)

    run_clustering(args.numpy_dir, n_components, args.n_clusters, args.results_file)
    sort_into_folders(args.bmp_dir, args.output_dir, args.results_file)


if __name__ == "__main__":
    main()
