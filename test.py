import augmentor
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2


def get_directory_contents(directory_path):
    """
    Gets the file paths of all the items in a directory.

    :param directory_path: Path to the directory.
    :return: The list of the absolute file paths to the files in a directory.
    """
    files = os.listdir(directory_path)
    abs_paths = [os.path.abspath(directory_path + "/" + file) for file in files]

    return abs_paths


def test_load(image_file_paths):
    """
    Test method of the load method of the augmentor class. Loads all images in a directory.

    :param image_file_paths: List of file paths.
    :return: list of cv2 image objects.
    """
    images = []
    for image_file_path in image_file_paths:
        image = augmentor.load(image_file_path)
        images.append(image)

    return images


def test_get_corners(images):
    """
    Test method for the _get_corners method of the augmentor class. gets the coordinates of the corners of the images.

    :param images: The images to get the corners of.
    :return: list of the corners of the images
    """
    corners = []
    for image in images:
        corners.append(augmentor._get_corners(image))

    return corners


def test__get_rand_points():
    """
    Test method for the _get_rand_points method of the augmentor class.

    :return: List of random points.
    """
    points = []
    for i in range(3):
        point = augmentor._get_rand_points(low=i, high=i + 1, shape=[i + 1, i + 1])
        points.append(point)

    return points


def test_perspective_transformation(images):
    """
    Test method for the perspective_transformation method of the augmentor class.

    :param images: list of images to transform.
    :return: List of cv2 image objects with a perspective transformation performed.
    """
    transformations = []
    kernel = np.array([[1, 0, 300], [0, 1, 150], [0, 0, 1]]).astype('float32')
    for image in images:
        transformed = augmentor.perspective_transformation(image, kernel)
        transformations.append(transformed)

    return transformations


def test_warp_corners(images):
    """
    Test method for the warp_corners method of the augmentor class.

    :param images: The images to transform.
    :return: List of cv2 image objects with the corners warped.
    """
    transformations = []
    for image in images:
        transformed = augmentor.warp_corners(image, augmentor._get_rand_points(0, 0.25, [4]),
                                             augmentor._get_rand_points(0, 0.25, [4]))
        transformations.append(transformed)

    return transformations


def test_horizontal_squeeze(images):
    """
    Test method for the horizontal_squeeze method of the augmentor class.

    :param images: The images to transform.
    :return: List of cv2 image objects with the corners warped in the horizontal direction.
    """
    transformations = []
    for image in images:
        transformed = augmentor.horizontal_squeeze(image, augmentor._get_rand_points(0, 0.25, [4]))
        transformations.append(transformed)

    return transformations


def test_vertical_squeeze(images):
    """
    Test method for the vertical_squeeze method of the augmentor class.

    :param images: The images to transform.
    :return: List of cv2 image objects with the corners warped in the vertical direction.
    """
    transformations = []
    for image in images:
        transformed = augmentor.vertical_squeeze(image, augmentor._get_rand_points(0, 0.25, [4]))
        transformations.append(transformed)

    return transformations


def test_affine_transformation(images):
    """
    Test method for the affine_transformation method of the augmentor class.

    :param images: The images to transform.
    :return: Images with the affine transformation
    """


#####################################
# Helper formatting methods
#####################################

def show_images(images):
    for image in images:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()


def announcment(method):
    annc = "# Testing {}".format(method)
    print("#" * len(annc))
    print(annc)
    print("#" * len(annc), end="\n\n")


def get_images(path):
    image_file_paths = get_directory_contents(path)
    # Getting image paths
    print("found the images:")
    for path in image_file_paths:
        print("\t{}".format(path))

    print()

    return image_file_paths


def load_images(image_file_paths):
    announcment("load")
    print("Loading the images...")
    images = test_load(image_file_paths)
    print("Done.", end="\n\n")

    return images


def corners(images):
    announcment("_get_corners")
    corners = test_get_corners(images)
    for corner in corners:
        print("found corner:\n{}".format(corner), end="\n\n")


def points():
    announcment("_get_rand_points")
    points = test__get_rand_points()
    for point in points:
        print("Created point:\n{}".format(point), end="\n\n")


def perspective_transform(images, show=True):
    announcment("perspective_transformation")
    images = test_perspective_transformation(images)
    if show:
        show_images(images)


def warp(images, show=True):
    announcment("warp_corners")
    images = test_warp_corners(images)
    if show:
        show_images(images)


def horizontal(images, show=True):
    announcment("horizontal_squeeze")
    images = test_horizontal_squeeze(images)
    if show:
        show_images(images)


def vertical(images, show=True):
    announcment("vertical_squeeze")
    images = test_vertical_squeeze(images)
    if show:
        show_images(images)


if __name__ == '__main__':
    paths = get_images("./images")
    images = load_images(paths)

    #######
    # TESTS
    #######

    # corners(images)
    # points()
    # perspective_transform(images)
    # warp(images)
    # horizontal(images)
    # vertical(images)
