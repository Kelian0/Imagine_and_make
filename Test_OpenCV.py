import numpy as np
import cv2

# Taille des cases de l'échiquier en mm
square_size = 6.75

# Dimensions de l'échiquier (nombre de coins en largeur et en hauteur)
chessboard_size = (8, 8)

# Liste pour stocker les coins détectés dans les images
image_points = []
object_points = []

image_paths = [
    "photo_calibration/1.jpg",
]

# Boucle sur les images de calibration
for image_path in image_paths:
    # Chargez l'image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tentez de détecter les coins de l'échiquier
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # Si les coins sont trouvés, ajoutez-les à la liste
        image_points.append(corners)

        # Générez les coordonnées 3D réelles des coins de l'échiquier
        object_points.append(np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32))
        object_points[-1][:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Maintenant, effectuez la calibration de la caméra
ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# Les résultats de calibration sont maintenant dans camera_matrix et distortion_coefficients
np.savez('camera_calibration.npz', camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients)