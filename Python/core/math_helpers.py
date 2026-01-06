"""
Math Helpers for Drone Simulation
Quaternion operations and vector mathematics.
"""

import numpy as np


def normalize_quaternion(q):
    """
    Normalize a quaternion to unit length.
    
    Args:
        q: Quaternion as [w, x, y, z]
    
    Returns:
        Normalized quaternion
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (Hamilton product).
    
    Args:
        q1, q2: Quaternions as [w, x, y, z]
    
    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_conjugate(q):
    """
    Return the conjugate of a quaternion.
    
    Args:
        q: Quaternion as [w, x, y, z]
    
    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate_vector_by_quaternion(vector, quaternion):
    """
    Rotate a 3D vector by a quaternion.
    Uses the formula: v' = q * v * q^(-1)
    
    Args:
        vector: 3D vector [x, y, z]
        quaternion: Unit quaternion [w, x, y, z]
    
    Returns:
        Rotated 3D vector [x', y', z']
    """
    # Normalize the quaternion to ensure it's a unit quaternion
    q = normalize_quaternion(quaternion)
    
    # Convert vector to pure quaternion [0, x, y, z]
    v_quat = np.array([0.0, vector[0], vector[1], vector[2]])
    
    # Compute q * v * q^(-1)
    # For unit quaternions, inverse = conjugate
    q_conj = quaternion_conjugate(q)
    
    # First: q * v
    temp = quaternion_multiply(q, v_quat)
    
    # Then: (q * v) * q^(-1)
    result = quaternion_multiply(temp, q_conj)
    
    # Extract the vector part (ignore w component)
    return np.array([result[1], result[2], result[3]])


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (in radians) to quaternion.
    Uses ZYX convention (yaw, pitch, roll).
    
    Args:
        roll: Rotation around X-axis (radians)
        pitch: Rotation around Y-axis (radians)
        yaw: Rotation around Z-axis (radians)
    
    Returns:
        Quaternion [w, x, y, z]
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def quaternion_to_euler(q):
    """
    Convert quaternion to Euler angles (in radians).
    Returns roll, pitch, yaw (ZYX convention).
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Tuple (roll, pitch, yaw) in radians
    """
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def quaternion_from_axis_angle(axis, angle):
    """
    Create a quaternion from an axis-angle representation.
    
    Args:
        axis: 3D unit vector [x, y, z] representing rotation axis
        angle: Rotation angle in radians
    
    Returns:
        Quaternion [w, x, y, z]
    """
    axis = np.array(axis) / np.linalg.norm(axis)  # Normalize axis
    half_angle = angle * 0.5
    sin_half = np.sin(half_angle)
    
    return np.array([
        np.cos(half_angle),
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half
    ])
