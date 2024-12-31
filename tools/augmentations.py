import numpy as np

def random_rotation(point_cloud, axis='z'):
    angle = np.random.rand() * 2 * np.pi  # Random angle in radians
    if axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    elif axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    return np.dot(point_cloud, rotation_matrix.T)

def random_scaling(point_cloud, scale_range=(0.8, 1.2)):
    scale = np.random.uniform(*scale_range)
    return point_cloud * scale

def random_translation(point_cloud, translate_range=(-0.2, 0.2)):
    translation = np.random.uniform(*translate_range, size=(3,))
    return point_cloud + translation

def random_jittering(point_cloud, sigma=0.01, clip=0.05):
    noise = np.clip(sigma * np.random.randn(*point_cloud.shape), -clip, clip)
    return point_cloud + noise

def random_dropout(point_cloud, drop_rate=0.1):
    keep_prob = 1 - drop_rate
    mask = np.random.rand(point_cloud.shape[0]) < keep_prob
    return point_cloud[mask]

def random_flip(point_cloud, axis='x'):
    if axis == 'x':
        point_cloud[:, 0] *= -1
    elif axis == 'y':
        point_cloud[:, 1] *= -1
    elif axis == 'z':
        point_cloud[:, 2] *= -1
    return point_cloud

def random_shear(point_cloud, shear_range=(-0.3, 0.3)):
    shear_factors = np.random.uniform(*shear_range, size=(2,))
    shear_matrix = np.array([
        [1, shear_factors[0], 0],
        [shear_factors[1], 1, 0],
        [0, 0, 1]
    ])
    return np.dot(point_cloud, shear_matrix.T)
