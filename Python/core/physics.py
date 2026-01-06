"""
Physics Engine for Drone Simulation
Handles physics updates, wind simulation, and force calculations.
"""

import numpy as np
from noise import pnoise3
from .math_helpers import rotate_vector_by_quaternion


class PhysicsEngine:
    """
    Core physics engine for drone simulation.
    Supports both fixed and variable timestep modes.
    """
    
    def __init__(self, dt=0.01, mode='fixed'):
        """
        Initialize the physics engine.
        
        Args:
            dt: Base timestep in seconds (default 0.01 = 100Hz)
            mode: 'fixed' or 'variable' timestep mode
        """
        self.dt = dt
        self.mode = mode
        self.gravity = np.array([0.0, 0.0, -9.81])
        
        # Accumulator for fixed timestep with variable frame times
        self.accumulator = 0.0
        
        # Wind simulation parameters
        self.wind_enabled = True
        self.wind_base_velocity = np.array([0.0, 0.0, 0.0])  # Base wind vector (m/s)
        self.wind_turbulence_scale = 0.05  # Spatial scale of turbulence
        self.wind_turbulence_strength = 3.0  # Max turbulence variation (m/s)
        self.wind_time_scale = 0.1  # How fast wind patterns evolve
    
    def step(self, drone, motor_speeds, sim_time):
        """
        Perform one physics step for a drone.
        
        Args:
            drone: DroneEntity object
            motor_speeds: Array of 4 motor speeds (RPM or rad/s)
            sim_time: Current simulation time
        """
        # A. Get Local Forces from Motors
        local_thrust_value = self._calculate_total_thrust(drone, motor_speeds)
        local_thrust_vector = np.array([0.0, 0.0, local_thrust_value])
        
        # B. Rotate Local Thrust to Global Space
        global_thrust = rotate_vector_by_quaternion(local_thrust_vector, drone.orientation)
        
        # C. Calculate Environmental Forces
        f_env = self._calculate_environmental_forces(drone, sim_time)
        
        # D. Net Force
        net_force = global_thrust + f_env
        
        # E. Update Linear Physics (Newton's 2nd Law: a = F/m)
        acceleration = net_force / drone.mass
        drone.acceleration = acceleration
        drone.velocity += acceleration * self.dt
        drone.position += drone.velocity * self.dt
        
        # F. Update Angular Physics
        self._update_angular_physics(drone, motor_speeds)
        
        # G. Update Battery
        drone.update_battery(local_thrust_value, self.dt)
    
    def update_with_accumulator(self, drones, motor_speeds_list, sim_time, frame_time):
        """
        Fixed timestep update with frame time accumulator.
        Ensures consistent physics regardless of frame rate.
        
        Args:
            drones: List of DroneEntity objects
            motor_speeds_list: List of motor speed arrays for each drone
            sim_time: Current simulation time
            frame_time: Actual elapsed time since last frame
        
        Returns:
            Number of physics steps taken
        """
        self.accumulator += frame_time
        steps = 0
        
        while self.accumulator >= self.dt:
            for drone, motor_speeds in zip(drones, motor_speeds_list):
                self.step(drone, motor_speeds, sim_time + steps * self.dt)
            self.accumulator -= self.dt
            steps += 1
        
        return steps
    
    def _calculate_total_thrust(self, drone, motor_speeds):
        """
        Calculate total thrust from motor speeds.
        
        Thrust = k * ω² (motor constant * angular velocity squared)
        """
        # Convert RPM to rad/s if needed (assuming input is RPM)
        omega = motor_speeds * (2 * np.pi / 60)
        
        # Calculate thrust from each motor
        thrust_per_motor = drone.motor_constant * omega ** 2
        
        return np.sum(thrust_per_motor)
    
    def _calculate_environmental_forces(self, drone, sim_time):
        """
        Calculate all environmental forces acting on the drone.
        
        Returns:
            Total environmental force vector [Fx, Fy, Fz]
        """
        # 1. Gravity Force (Weight = mass * g)
        f_gravity = drone.mass * self.gravity
        
        # 2. Wind Force
        f_wind = np.zeros(3)
        if self.wind_enabled:
            wind_velocity = self.get_wind_at_position(drone.position, sim_time)
            # Simplified drag: F = 0.5 * ρ * Cd * A * v²
            # Using simplified coefficient for drone body
            relative_wind = wind_velocity - drone.velocity
            drag_coefficient = 0.5  # Approximate for drone body
            reference_area = 0.1  # m² approximate frontal area
            air_density = 1.225  # kg/m³ at sea level
            
            wind_speed = np.linalg.norm(relative_wind)
            if wind_speed > 0.01:
                wind_direction = relative_wind / wind_speed
                f_wind = 0.5 * air_density * drag_coefficient * reference_area * wind_speed ** 2 * wind_direction
        
        return f_gravity + f_wind
    
    def get_wind_at_position(self, position, sim_time):
        """
        Calculate wind vector at a specific position using Perlin noise.
        Creates smooth, continuous turbulence that varies spatially and temporally.
        
        Args:
            position: [x, y, z] position in meters
            sim_time: Current simulation time
        
        Returns:
            Wind velocity vector [vx, vy, vz] in m/s
        """
        scale = self.wind_turbulence_scale
        time_input = sim_time * self.wind_time_scale
        
        # Sample Perlin noise at slightly offset coordinates for each axis
        # This creates different but correlated turbulence in each direction
        wx = pnoise3(
            position[0] * scale,
            position[1] * scale,
            time_input,
            octaves=4,
            persistence=0.5,
            lacunarity=2.0
        )
        
        wy = pnoise3(
            position[0] * scale + 100,  # Offset to decorrelate
            position[1] * scale + 100,
            time_input,
            octaves=4,
            persistence=0.5,
            lacunarity=2.0
        )
        
        wz = pnoise3(
            position[0] * scale + 200,
            position[1] * scale + 200,
            time_input + 100,
            octaves=4,
            persistence=0.5,
            lacunarity=2.0
        )
        
        # Scale turbulence and add to base wind
        turbulence = np.array([wx, wy, wz]) * self.wind_turbulence_strength
        
        # Reduce vertical turbulence (more realistic)
        turbulence[2] *= 0.3
        
        return self.wind_base_velocity + turbulence
    
    def _update_angular_physics(self, drone, motor_speeds):
        """
        Update drone orientation based on motor differentials.
        Uses quaternion integration for rotation.
        """
        # Calculate torques from motor speed differentials
        # For a quadcopter in X configuration:
        # - Roll (x-axis): difference between left and right motors
        # - Pitch (y-axis): difference between front and back motors
        # - Yaw (z-axis): difference between CW and CCW motors
        
        omega = motor_speeds * (2 * np.pi / 60)
        thrust_per_motor = drone.motor_constant * omega ** 2
        
        # Simplified torque calculation (assuming X configuration)
        L = drone.arm_length
        tau_roll = L * (thrust_per_motor[1] + thrust_per_motor[2] - thrust_per_motor[0] - thrust_per_motor[3])
        tau_pitch = L * (thrust_per_motor[0] + thrust_per_motor[1] - thrust_per_motor[2] - thrust_per_motor[3])
        
        # Yaw torque from reactive torque (simplified)
        yaw_constant = 0.01
        tau_yaw = yaw_constant * (omega[0] ** 2 - omega[1] ** 2 + omega[2] ** 2 - omega[3] ** 2)
        
        torques = np.array([tau_roll, tau_pitch, tau_yaw])
        
        # Angular acceleration (τ = I * α)
        angular_acceleration = torques / drone.moment_of_inertia
        
        # Update angular velocity
        drone.angular_velocity += angular_acceleration * self.dt
        
        # Update orientation quaternion
        self._integrate_quaternion(drone)
    
    def _integrate_quaternion(self, drone):
        """
        Integrate angular velocity into orientation quaternion.
        Uses first-order integration (sufficient for small timesteps).
        """
        omega = drone.angular_velocity
        q = drone.orientation
        
        # Quaternion derivative: q_dot = 0.5 * q * omega_quat
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        
        # Hamilton product for q * omega_quat
        w, x, y, z = q
        ow, ox, oy, oz = omega_quat
        
        q_dot = 0.5 * np.array([
            w * ow - x * ox - y * oy - z * oz,
            w * ox + x * ow + y * oz - z * oy,
            w * oy - x * oz + y * ow + z * ox,
            w * oz + x * oy - y * ox + z * ow
        ])
        
        # Integrate
        drone.orientation = q + q_dot * self.dt
        
        # Renormalize to prevent drift
        drone.orientation = drone.orientation / np.linalg.norm(drone.orientation)
    
    def set_wind_conditions(self, base_velocity=None, turbulence_strength=None, enabled=None):
        """
        Configure wind simulation parameters.
        
        Args:
            base_velocity: [vx, vy, vz] base wind velocity in m/s
            turbulence_strength: Max turbulence variation in m/s
            enabled: Enable/disable wind simulation
        """
        if base_velocity is not None:
            self.wind_base_velocity = np.array(base_velocity)
        if turbulence_strength is not None:
            self.wind_turbulence_strength = turbulence_strength
        if enabled is not None:
            self.wind_enabled = enabled


class TimestepManager:
    """
    Manages simulation timestep, supporting both fixed and variable modes.
    """
    
    def __init__(self, target_dt=0.01, mode='fixed', max_frame_time=0.1):
        """
        Initialize timestep manager.
        
        Args:
            target_dt: Target timestep in seconds
            mode: 'fixed', 'variable', or 'semi-fixed'
            max_frame_time: Maximum allowed frame time (prevents spiral of death)
        """
        self.target_dt = target_dt
        self.mode = mode
        self.max_frame_time = max_frame_time
        
        self.accumulator = 0.0
        self.sim_time = 0.0
        self.real_time = 0.0
        self.frame_count = 0
    
    def get_timesteps(self, elapsed_real_time):
        """
        Get the timesteps to execute based on elapsed real time.
        
        Args:
            elapsed_real_time: Time since last frame in seconds
        
        Returns:
            List of (dt, sim_time) tuples for each step to execute
        """
        # Clamp to prevent spiral of death
        elapsed = min(elapsed_real_time, self.max_frame_time)
        
        if self.mode == 'fixed':
            # Fixed timestep with accumulator
            self.accumulator += elapsed
            steps = []
            
            while self.accumulator >= self.target_dt:
                steps.append((self.target_dt, self.sim_time))
                self.sim_time += self.target_dt
                self.accumulator -= self.target_dt
            
            return steps
        
        elif self.mode == 'variable':
            # Single variable timestep
            self.sim_time += elapsed
            return [(elapsed, self.sim_time - elapsed)]
        
        else:  # semi-fixed
            # Variable number of fixed steps
            self.accumulator += elapsed
            steps = []
            
            # Take multiple fixed steps if needed
            while self.accumulator >= self.target_dt:
                steps.append((self.target_dt, self.sim_time))
                self.sim_time += self.target_dt
                self.accumulator -= self.target_dt
            
            return steps
    
    def get_interpolation_alpha(self):
        """
        Get interpolation factor for rendering between physics states.
        Useful for smooth rendering when physics rate != render rate.
        
        Returns:
            Alpha value between 0 and 1
        """
        return self.accumulator / self.target_dt
