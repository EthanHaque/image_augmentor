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
    :return: List of cv2 image objects with the affine transformation
    """
    transformations = []
    kernel = np.array([[0.9, 0, 10], [0, 0.75, 20]]).astype('float32')
    for image in images:
        transformed = augmentor.affine_transformation(image, kernel)
        transformations.append(transformed)

    return transformations


def test_translate(images):
    """
    Test method for the translate method of the augmentor class.

    :param images: The images to transform.
    :return: List of cv2 image objects with the translation.
    """
    transformations = []
    for image in images:
        transformed = augmentor.translate(image, 200, 200)
        transformations.append(transformed)

    return transformations


def test_rotate(images):
    """
    Test method for the rotate method of the augmentor class.

    :param images: The images to transform.
    :return: List of cv2 image objects with the rotation.
    """
    transformations = []
    for image in images:
        transformed = augmentor.rotate(image, 10)
        transformations.append(transformed)

    return transformations


def test_scale(images):
    """
    Test method for the scale method of the augmentor class.

    :param images: The images to transform.
    :return: List of cv2 image objects with the scaling.
    """
    transformations = []
    for image in images:
        transformed = augmentor.scale(image, 0.25, 0.25)
        transformations.append(transformed)

    return transformations


def test_average_blur(images):
    """
    Test method for the average_blur method of the augmentor class.

    :param images: The images to transform.
    :return: List of cv2 image objects with the average blur applied.
    """
    transformations = []
    for image in images:
        transformed = augmentor.average_blur(image, 10)
        transformations.append(transformed)

    return transformations


def test_median_blur(images):
    """
    Test method for the median_blur method of the augmentor class.

    :param images: The images to transform.
    :return: List of cv2 image objects with the median blur applied.
    """
    transformations = []
    for image in images:
        transformed = augmentor.median_blur(image, 3)
        transformations.append(transformed)

    return transformations


def test_gaussian_blur(images):
    """
    Test method for the gaussian_blur method of the augmentor class.

    :param images: The images to transform.
    :return: List of cv2 image objects with the gaussian blur applied.
    """
    transformations = []
    for image in images:
        transformed = augmentor.gaussian_blur(image, 2, 2)
        transformations.append(transformed)

    return transformations


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


def affine(images, show=True):
    announcment("affine_transformation")
    images = test_affine_transformation(images)
    if show:
        show_images(images)


def translate(images, show=True):
    announcment("translate")
    images = test_translate(images)
    if show:
        show_images(images)


def rotate(images, show=True):
    announcment("rotate")
    images = test_rotate(images)
    if show:
        show_images(images)


def scale(images, show=True):
    announcment("scale")
    images = test_scale(images)
    if show:
        show_images(images)


def average_blur(images, show=True):
    announcment("average_blur")
    images = test_average_blur(images)
    if show:
        show_images(images)


def median_blur(images, show=True):
    announcment("test_median_blur")
    images = test_median_blur(images)
    if show:
        show_images(images)

def gaussian_blur(images, show=True):
    announcment("gaussian_blur")
    images = test_gaussian_blur(images)
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
    # affine(images)
    # translate(images)
    # rotate(images)
    # scale(images)
    # average_blur(images)
    # median_blur(images)
    gaussian_blur(images)
