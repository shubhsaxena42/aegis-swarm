"""
Drone Entity Module - Represents a single drone with full state vector.
"""
import numpy as np


class DroneEntity:
    def __init__(self, drone_id, initial_pos, initial_orientation=None):
        self.id = drone_id
        
        # State Vectors (x, y, z)
        self.position = np.array(initial_pos, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.acceleration = np.zeros(3, dtype=np.float64)
        self.angular_velocity = np.zeros(3, dtype=np.float64)
        
        # Orientation (Quaternion: [w, x, y, z])
        if initial_orientation is not None:
            self.orientation = np.array(initial_orientation, dtype=np.float64)
        else:
            self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Physical Properties
        self.mass = 1.5  # kg
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.arm_length = 0.25  # meters
        self.motor_constant = 8.54e-06
        self.moment_of_inertia = np.array([0.015, 0.015, 0.02])
        
        # Battery State
        self.battery_capacity_ah = 5.0
        self.battery_level = 100.0
        self.battery_voltage_nominal = 22.2
        self.battery_voltage_current = 22.2
        self.total_energy_consumed = 0.0
        
        # Motor State
        self.motor_speeds = np.zeros(4)
        self.motor_max_rpm = 10000

    def update_battery(self, thrust_magnitude, dt):
        """C-rate dependent battery discharge with Peukert effect."""
        if thrust_magnitude <= 0:
            return
        
        k_current = 0.05
        current_draw = k_current * (thrust_magnitude ** 1.5)
        c_rate = current_draw / self.battery_capacity_ah
        
        # Voltage sag
        internal_resistance = 0.02
        self.battery_voltage_current = self.battery_voltage_nominal - (current_draw * internal_resistance) - (0.05 * c_rate)
        self.battery_voltage_current = max(self.battery_voltage_current, 18.0)
        
        # Energy consumption
        power_draw = self.battery_voltage_current * current_draw
        self.total_energy_consumed += (power_draw * dt) / 3600
        
        # Peukert discharge
        nominal_discharge = (current_draw * dt) / (self.battery_capacity_ah * 3600) * 100
        peukert_penalty = (c_rate ** 0.15) if c_rate > 1.0 else 1.0
        self.battery_level -= nominal_discharge * peukert_penalty
        self.battery_level = max(0.0, min(100.0, self.battery_level))

    def get_state_vector(self):
        return np.concatenate([self.position, self.velocity, self.acceleration, 
                               self.orientation, self.angular_velocity])

    def get_euler_angles(self):
        w, x, y, z = self.orientation
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        sinp = 2*(w*y - z*x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw

    def get_hover_motor_speed(self):
        thrust_per_motor = (self.mass * 9.81) / 4
        omega = np.sqrt(thrust_per_motor / self.motor_constant)
        return omega * 60 / (2 * np.pi)

    def __repr__(self):
        return f"DroneEntity(id={self.id}, pos={self.position}, battery={self.battery_level:.1f}%)"