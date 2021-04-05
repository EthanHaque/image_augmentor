import augmentor
import os


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
    out = []
    for image in images:
        out.append(augmentor._get_corners(image))

    return out


#####################################
# Helper formatting methods
#####################################

def get_images(path):
    image_file_paths = get_directory_contents(path)
    # Getting image paths
    print("found the images:")
    for path in image_file_paths:
        print("\t{}".format(path))

    return image_file_paths


def load_images(image_file_paths):
    print()
    print("Loading the images...")
    images = test_load(image_file_paths)
    print("Done", end="\n\n")

    return images


def corners(images):
    corners = test_get_corners(images)
    for corner in corners:
        print("found corner:\n{}".format(corner), end="\n\n")


if __name__ == '__main__':
    paths = get_images("./images")
    images = load_images(paths)

    #######
    # TESTS
    #######
    print("############")
    print("# PERFORMING TESTS")
    print("############", end="\n\n")

    corners(images)
