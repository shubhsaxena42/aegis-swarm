import time
import numpy as np
from core.drone_entity import DroneEntity
from core.physics import PhysicsEngine, TimestepManager


def create_swarm(num_drones, spacing=2.0):
    """Create a grid formation of drones."""
    swarm = []
    side_length = int(np.ceil(np.sqrt(num_drones)))
    
    for i in range(num_drones):
        row = i // side_length
        col = i % side_length
        pos = [col * spacing, row * spacing, 0.0]
        drone = DroneEntity(drone_id=i, initial_pos=pos)
        swarm.append(drone)
    return swarm


def main():
    # === INITIALIZATION ===
    dt = 0.01  # 100Hz Fixed Timestep
    physics = PhysicsEngine(dt=dt, mode='fixed')
    timestep_mgr = TimestepManager(target_dt=dt, mode='fixed')
    
    # Configure wind conditions
    physics.set_wind_conditions(
        base_velocity=[2.0, 1.0, 0.0],  # Light breeze from SW
        turbulence_strength=2.0,
        enabled=True
    )
    
    # Create drone swarm
    swarm = create_swarm(50)
    print(f"Created swarm of {len(swarm)} drones")
    print(f"Simulation timestep: {dt}s ({1/dt:.0f}Hz)")
    print(f"Wind enabled with turbulence")
    print("-" * 50)
    
    # === MAIN SIMULATION LOOP ===
    sim_time = 0.0
    last_print_time = 0.0
    
    try:
        while sim_time < 10.0:
            loop_start = time.time()
            
            for drone in swarm:
                # Get hover motor speeds (plus small adjustments for demo)
                hover_rpm = drone.get_hover_motor_speed()
                motor_speeds = np.array([hover_rpm] * 4)
                
                # Run physics step
                physics.step(drone, motor_speeds, sim_time)
            
            # Telemetry output every 0.5s
            if sim_time - last_print_time >= 0.5:
                d = swarm[0]
                wind = physics.get_wind_at_position(d.position, sim_time)
                print(f"T: {sim_time:5.2f}s | Pos: [{d.position[0]:6.2f}, {d.position[1]:6.2f}, {d.position[2]:6.2f}] | "
                      f"Bat: {d.battery_level:5.1f}% | Wind: [{wind[0]:4.1f}, {wind[1]:4.1f}] m/s")
                last_print_time = sim_time
            
            sim_time += dt
            
            # Real-time sync
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    
    # Final summary
    print("-" * 50)
    print("SIMULATION COMPLETE")
    d = swarm[0]
    print(f"Final position: {d.position}")
    print(f"Final battery: {d.battery_level:.1f}%")
    print(f"Energy consumed: {d.total_energy_consumed:.2f} Wh")


if __name__ == "__main__":
    main()