#!/usr/bin/env python

from __future__ import with_statement

import os
import shutil
import logging
from geopy.geocoders import Nominatim
import eyed3
import hashlib
import exifread
import sys
import errno
import time
import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from fuse import FUSE, FuseOSError, Operations
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the filesystem. The root directory for the filesystem is specified by `root`.
class Passthrough(Operations):
    def __init__(self, root):
        self.root = root

    """
    Helpers
    """
    # Generates the full path for the filesystem based on the partial path given. Concatenates the root directory with the partial path to get the complete path.
    def _full_path(self, partial):
        if partial.startswith("/"):
            partial = partial[1:]
        path = os.path.join(self.root, partial)
        return path

    """
    Filesystem methods
    """
    # Check file access and requested operation permissions. If not, it raises a FuseOSError exception.
    def access(self, path, mode):
        full_path = self._full_path(path)
        if not os.access(full_path, mode):
            raise FuseOSError(errno.EACCES)

    # Change the mode (permissions) of a file. The mode is a Unix mode bitfield.
    def chmod(self, path, mode):
        full_path = self._full_path(path)
        return os.chmod(full_path, mode)

    # Change the owner and group of a file. This method changes the owner (UID) and group (GID) of the file.
    def chown(self, path, uid, gid):
        full_path = self._full_path(path)
        return os.chown(full_path, uid, gid)

    # Returns a dictionary of attributes that describe the file or directory. E.g. size, modification time, etc.
    def getattr(self, path, fh=None):
        full_path = self._full_path(path)
        st = os.lstat(full_path)
        return dict((key, getattr(st, key)) for key in ('st_atime', 'st_ctime', 'st_gid', 'st_mode', 'st_mtime', 'st_nlink', 'st_size', 'st_uid'))

    # Read directory contents. This method returns a generator yielding the names of the entries in the directory.
    def readdir(self, path, fh):
        full_path = self._full_path(path)

        dirents = ['.', '..']
        if os.path.isdir(full_path):
            dirents.extend(os.listdir(full_path))
        for r in dirents:
            yield r

    # Read the target of a symbolic link. The return value is the path to which the symbolic link points.
    def readlink(self, path):
        pathname = os.readlink(self._full_path(path))
        if pathname.startswith("/"):
            # Path name is absolute, sanitize it.
            return os.path.relpath(pathname, self.root)
        else:
            return pathname

    # Create a filesystem node (file, device special file, or named pipe) named `path`.
    def mknod(self, path, mode, dev):
        return os.mknod(self._full_path(path), mode, dev)

    # Remove a directory. This method removes (deletes) a directory.
    def rmdir(self, path):
        full_path = self._full_path(path)
        return os.rmdir(full_path)
    
    # Create a directory named `path` with numeric mode `mode`.
    def mkdir(self, path, mode):
        return os.mkdir(self._full_path(path), mode)

    # Get filesystem statistics. Returns a dictionary with keys that are attributes like number of free blocks, etc.
    def statfs(self, path):
        full_path = self._full_path(path)
        stv = os.statvfs(full_path)
        return dict((key, getattr(stv, key)) for key in ('f_bavail', 'f_bfree', 'f_blocks', 'f_bsize', 'f_favail', 'f_ffree', 'f_files', 'f_flag', 'f_frsize', 'f_namemax'))

    # Remove (delete) a file. `path` is the file to remove.
    def unlink(self, path):
        return os.unlink(self._full_path(path))

    # Create a symbolic link `name` pointing to `target`.
    def symlink(self, name, target):
        return os.symlink(name, self._full_path(target))

    # Rename a file or directory from `old` to `new`.
    def rename(self, old, new):
        return os.rename(self._full_path(old), self._full_path(new))

    # Create a hard link pointing to `target` named `name`.
    def link(self, target, name):
        return os.link(self._full_path(target), self._full_path(name))

    # Set file times. `times` is a 2-tuple of the form (atime, mtime) where `atime` and `mtime` are the access and modification times, respectively.
    def utimens(self, path, times=None):
        return os.utime(self._full_path(path), times)

    """
    File methods
    """
    # Open a file. The `flags` are passed directly to the `os.open` method.
    def open(self, path, flags):
        full_path = self._full_path(path)
        return os.open(full_path, flags)

    # Create and open a file. The file is created with mode `mode` and opened.
    def create(self, path, mode, fi=None):
        full_path = self._full_path(path)
        return os.open(full_path, os.O_WRONLY | os.O_CREAT, mode)

    # Read from a file. Reads `length` bytes from the file descriptor `fh` starting at `offset`.
    def read(self, path, length, offset, fh):
        os.lseek(fh, offset, os.SEEK_SET)
        return os.read(fh, length)

    # Write to a file. Writes the buffer `buf` to the file descriptor `fh` starting at `offset`.
    def write(self, path, buf, offset, fh):
        os.lseek(fh, offset, os.SEEK_SET)
        return os.write(fh, buf)

    # Truncate a file to a specified length. If `fh` is not specified, `path` is used to open the file.
    def truncate(self, path, length, fh=None):
        full_path = self._full_path(path)
        with open(full_path, 'r+') as f:
            f.truncate(length)

    # Flush cached data. Ensures that changes made to a file are written to the storage device.
    def flush(self, path, fh):
        return os.fsync(fh)

    # Release an open file. Close the file descriptor `fh`.
    def release(self, path, fh):
        return os.close(fh)
    
    # Synchronize file contents. If `fdatasync` is true, only the file's data is flushed, not its metadata.
    def fsync(self, path, fdatasync, fh):
        return self.flush(path, fh)


# BASIC FILE SORTING METHODS


# Prints out file names in unsorted order
def print_file_names(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))

# Prints out file names in sorted order
def print_sorted_file_names(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    all_files.sort()
    for file in all_files:
        print(file)

# Prints out file access times in unsorted order
def print_file_access_times(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            atime = os.stat(full_path).st_atime
            print(f"{full_path}: Accessed on {time.ctime(atime)}")

# Prints out file access times in sorted order
def print_sorted_file_access_times(directory):
    files_with_atime = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            atime = os.stat(full_path).st_atime
            files_with_atime.append((full_path, atime))
    files_with_atime.sort(key=lambda x: x[1])
    for full_path, atime in files_with_atime:
        print(f"{full_path}: Accessed on {time.ctime(atime)}")

# Prints out file modification times in unsorted order
def print_file_modification_times(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            mtime = os.stat(full_path).st_mtime
            print(f"{full_path}: Modified on {time.ctime(mtime)}")

# Prints out file modification times in sorted order
def print_sorted_file_modification_times(directory):
    files_with_mtime = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            mtime = os.stat(full_path).st_mtime
            files_with_mtime.append((full_path, mtime))
    files_with_mtime.sort(key=lambda x: x[1])
    for full_path, mtime in files_with_mtime:
        print(f"{full_path}: Modified on {time.ctime(mtime)}")

# Prints out file change times in unsorted order
def print_file_change_times(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            ctime = os.stat(full_path).st_ctime
            print(f"{full_path}: Inode changed on {time.ctime(ctime)}")

# Prints out file change times in sorted order
def print_sorted_file_change_times(directory):
    files_with_ctime = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            ctime = os.stat(full_path).st_ctime
            files_with_ctime.append((full_path, ctime))
    files_with_ctime.sort(key=lambda x: x[1])
    for full_path, ctime in files_with_ctime:
        print(f"{full_path}: Inode changed on {time.ctime(ctime)}")

# Prints out file sizes in unsorted order
def print_file_sizes(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            size = os.stat(full_path).st_size
            print(f"{full_path}: Size = {size} bytes")

# Prints out file sizes in sorted order
def print_sorted_file_sizes(directory):
    files_with_size = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            size = os.stat(full_path).st_size
            files_with_size.append((full_path, size))
    files_with_size.sort(key=lambda x: x[1])
    for full_path, size in files_with_size:
        print(f"{full_path}: Size = {size} bytes")

# General method to sort and print files by a given metadata attribute.
def print_files_sorted_by_metadata(directory, attr):
    files_with_metadata = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            stat_info = os.stat(full_path)
            metadata_value = getattr(stat_info, attr, None)
            files_with_metadata.append((full_path, metadata_value))
    files_with_metadata.sort(key=lambda x: x[1])
    for full_path, metadata_value in files_with_metadata:
        print(f"{full_path}: {attr} = {metadata_value}")


# DUPLICATE FILE DETECTION


# Generate a hash for the metadata of a file.
def generate_metadata_hash(file_path):
    try:
        stat_info = os.stat(file_path)
        metadata = (stat_info.st_size, stat_info.st_mtime)

        hasher = hashlib.md5()
        hasher.update(repr(metadata).encode('utf-8'))
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Finds and prints paths of duplicate files based on file metadata.
def find_duplicate_files(directory):
    hashes = {}
    duplicates = []

    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_hash = generate_metadata_hash(file_path)

            if file_hash:
                if file_hash in hashes:
                    duplicates.append((file_path, hashes[file_hash]))
                else:
                    hashes[file_hash] = file_path

    if duplicates:
        print("Duplicate files found:")
        for dup in duplicates:
            print(f"{dup[0]} is a duplicate of {dup[1]}")
    else:
        print("No duplicate files found.")

# Finds and deletes paths of duplicate files based on file metadata.
def delete_duplicate_files(directory):
    hashes = {}
    duplicates = []

    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_hash = generate_metadata_hash(file_path)

            if file_hash:
                if file_hash in hashes:
                    duplicates.append(file_path) 
                else:
                    hashes[file_hash] = file_path

    for dup_path in duplicates:
        try:
            os.remove(dup_path)
            print(f"Removed duplicate file: {dup_path}")
        except Exception as e:
            print(f"Failed to remove {dup_path}: {e}")

    if not duplicates:
        print("No duplicate files found.")


# MUSIC GROUPING METHODS


def extract_and_group_mp3_metadata(directory):

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.mp3'):
                file_path = os.path.join(root, file)
                audiofile = eyed3.load(file_path)
                
                if audiofile is not None:
                    if audiofile.tag is not None:
                        artist = audiofile.tag.artist if audiofile.tag.artist else 'Unknown Artist'
                        album = audiofile.tag.album if audiofile.tag.album else 'Unknown Album'

                        # Define the directory path based on artist and album
                        group_directory = os.path.join(directory, sanitize_filename(artist), sanitize_filename(album))
                        os.makedirs(group_directory, exist_ok=True)

                        # Move the file to the new directory
                        new_file_path = os.path.join(group_directory, file)
                        shutil.move(file_path, new_file_path)
                        print(f"Moved {file_path} to {new_file_path}")
                    else:
                        print("No ID3 tag found for:", file_path)
                else:
                    print("Could not load the MP3 file:", file_path)

def group_mp3_by_metadata(directory, metadata_key):

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.mp3'):
                file_path = os.path.join(root, filename)
                audiofile = eyed3.load(file_path)
                
                if audiofile is not None and audiofile.tag is not None:
                    metadata_value = getattr(audiofile.tag, metadata_key, None)
                    if metadata_value:
                        metadata_value = str(metadata_value).strip()  # Convert to string and clean it
                    else:
                        metadata_value = 'No_Attribute'  # Assign a default group for missing metadata
                    
                    # Sanitize the metadata value to be used as a directory name
                    group_directory = os.path.join(directory, sanitize_filename(metadata_value))
                    os.makedirs(group_directory, exist_ok=True)  # Create a directory for the group if it doesn't exist

                    # Move the file to the new directory
                    new_file_path = os.path.join(group_directory, filename)
                    shutil.move(file_path, new_file_path)
                    print(f"Moved {file_path} to {new_file_path}")
                else:
                    print(f"Could not load ID3 tags for file: {file_path}")

def sanitize_filename(name):
    
    return "".join(c for c in name if c.isalnum() or c in " -_").rstrip()


# PHOTO GROUPING METHODS


# Extract datetime from EXIF data.
def get_exif_datetime(tags):
    exif_time = tags.get('EXIF DateTimeOriginal')
    if exif_time:
        return datetime.datetime.strptime(str(exif_time), '%Y:%m:%d %H:%M:%S')
    return None

#  Extract GPS coordinates from EXIF data if available
def get_gps_coordinates(tags):
    gps_latitude = tags.get('GPS GPSLatitude')
    gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
    gps_longitude = tags.get('GPS GPSLongitude')
    gps_longitude_ref = tags.get('GPS GPSLongitudeRef')

    if not all([gps_latitude, gps_latitude_ref, gps_longitude, gps_longitude_ref]):
        return None  # GPS information not available

    def convert_to_degrees(value):
        """ Convert GPS coordinates to degrees """
        d, m, s = value.values
        return d.num / d.den + (m.num / m.den / 60.0) + (s.num / s.den / 3600.0)

    latitude = convert_to_degrees(gps_latitude)
    if gps_latitude_ref.values != 'N':
        latitude = -latitude
    longitude = convert_to_degrees(gps_longitude)
    if gps_longitude_ref.values != 'E':
        longitude = -longitude

    return (latitude, longitude)

# Prepare data for clustering, including paths for re-organization.
def prepare_image_data(directory):
    locations_times = []
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    tags = exifread.process_file(f, details=False)
                    location = get_gps_coordinates(tags)
                    time_taken = get_exif_datetime(tags)
                    if location and time_taken:
                        timestamp = time_taken.timestamp()
                        locations_times.append([location[0], location[1], timestamp])
                        paths.append(file_path)
    return locations_times, paths

# Run K-Means clustering on the prepared data.
def run_kmeans_clustering(data, n_clusters=4, n_init=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

# Use geopy to get country name from latitude and longitude.
def reverse_geocode(latitude, longitude):
    geolocator = Nominatim(user_agent="my_unique_geocoder")
    try:
        location = geolocator.reverse((latitude, longitude), exactly_one=True)
        if location is not None:
            address = location.raw.get('address', {})
            country = address.get('country', 'Unknown')
            return country
        else:
            return "Unknown"
    except Exception as e:
        print(f"Error in geocoding: {e}")
        return "Unknown"

# Cluster JPEG images and organize them into directories based on location and time.
def cluster_images_by_location_and_time_with_known_k(directory, n_clusters=4):
    data, paths = prepare_image_data(directory)
    if data:
        labels, centers = run_kmeans_clustering(data, n_clusters)
        for i, label in enumerate(labels):
            image_path = paths[i]
            if os.path.exists(image_path):  # Check if the file exists
                try:
                    country = reverse_geocode(centers[label][0], centers[label][1])
                    year = datetime.datetime.fromtimestamp(centers[label][2]).year
                    cluster_dir = os.path.join(directory, f"{country}_{year}")
                    os.makedirs(cluster_dir, exist_ok=True)
                    
                    new_file_path = os.path.join(cluster_dir, os.path.basename(image_path))
                    shutil.move(image_path, new_file_path)
                    print(f"Moved {image_path} to {new_file_path}")
                except Exception as e:
                    logging.error(f"Error moving file {image_path} to {cluster_dir}: {e}")
            else:
                logging.warning(f"File not found: {image_path}")
    else:
        print("No sufficient data to perform clustering.")

# Finds optimal k with silhouette score
def determine_optimal_k(data, max_k=10):
    scores = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(data)
        if k > 1:
            score = silhouette_score(data, kmeans.labels_)
            scores.append(score)
    optimal_k = scores.index(max(scores)) + 2
    return optimal_k

# Cluster JPEG images and organize them into directories based on location and time.
def cluster_images_by_location_and_time_without_known_k(directory):
    data, paths = prepare_image_data(directory)
    if len(data) == 0:
        print("No data available to perform clustering.")
        return

    optimal_k = determine_optimal_k(data)
    print(f"Optimal number of clusters determined: {optimal_k}")
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)

    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    for i, label in enumerate(labels):
        image_path = paths[i]
        country = reverse_geocode(centers[label][0], centers[label][1])
        year = datetime.datetime.fromtimestamp(centers[label][2]).year
        cluster_dir = os.path.join(directory, f"{country}_{year}")
        os.makedirs(cluster_dir, exist_ok=True)
        new_file_path = os.path.join(cluster_dir, os.path.basename(image_path))
        shutil.move(image_path, new_file_path)
        print(f"Moved {image_path} to {new_file_path}")


# TEXT RECOGNITION GROUPING TF-IDF

# Extract all text from a PDF file.
def extract_text_from_pdf_tf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + " "
    except Exception as e:
        print(f"Failed to extract text from {pdf_path}: {e}")
    return text.strip()

# Preprocess text by lowercasing and removing punctuation.
def preprocess_text_tf(text):
    import re
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Cluster PDF documents based on text content similarity.
def cluster_pdfs_tf(directory, n_clusters=5):
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    texts = [extract_text_from_pdf_tf(pdf) for pdf in pdf_files]
    texts = [preprocess_text_tf(text) for text in texts]

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(tfidf_matrix)
    labels = kmeans.labels_

    for label, pdf_file in zip(labels, pdf_files):
        group_dir = os.path.join(directory, f"Cluster_{label}")
        os.makedirs(group_dir, exist_ok=True)
        shutil.move(pdf_file, os.path.join(group_dir, os.path.basename(pdf_file)))

    print(f"PDF files have been grouped into {n_clusters} clusters based on their text content.")


# MISC FILE MANAGEMENT FUNCTIONS

# Move all files from subdirectories into the main directory, leaving them as loose files.
def flatten_directory_structure(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            new_path = os.path.join(directory, name)

            if file_path != new_path:
                os.rename(file_path, new_path)
                print(f"Moved {file_path} to {new_path}")

    for root, dirs, files in os.walk(directory, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")
            except OSError as e:
                print(f"Error removing directory {dir_path}: {e}")

# Print metadata of PNG files found in the specified directory.
def print_png_metadata(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                with Image.open(full_path) as img:
                    metadata = img.info
                    print(f"Metadata for {file}: {metadata}")


def main(mountpoint, root):

    # flatten_directory_structure(root)
    # cluster_images_by_location_and_time_without_known_k(root)
    cluster_pdfs_tf(root, n_clusters=3)
    FUSE(Passthrough(root), mountpoint, nothreads=True, foreground=True)

if __name__ == '__main__':
    main(sys.argv[2], sys.argv[1])