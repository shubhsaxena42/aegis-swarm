"""
Coordinate System Utilities
MGRS (Military Grid Reference System) and UTM (Universal Transverse Mercator) conversions.
"""

import numpy as np
import re


# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis (equatorial radius) in meters
WGS84_B = 6356752.314245  # Semi-minor axis (polar radius) in meters
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_E = np.sqrt(1 - (WGS84_B / WGS84_A) ** 2)  # First eccentricity
WGS84_E2 = WGS84_E ** 2  # Eccentricity squared

# UTM Scale factor
UTM_K0 = 0.9996

# MGRS latitude band letters (C to X, excluding I and O)
MGRS_LATITUDE_BANDS = "CDEFGHJKLMNPQRSTUVWX"

# 100km grid square column letters (A-Z excluding I and O, repeats every 8)
MGRS_COLUMN_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

# 100km grid square row letters (A-V excluding I and O, repeats every 20)
MGRS_ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"


class UTMCoordinate:
    """Represents a UTM coordinate."""
    
    def __init__(self, easting, northing, zone_number, zone_letter, hemisphere='N'):
        self.easting = easting
        self.northing = northing
        self.zone_number = zone_number
        self.zone_letter = zone_letter
        self.hemisphere = hemisphere
    
    def __repr__(self):
        return f"UTM({self.zone_number}{self.zone_letter} {self.easting:.2f}E {self.northing:.2f}N)"
    
    def to_tuple(self):
        return (self.easting, self.northing, self.zone_number, self.zone_letter)


class MGRSCoordinate:
    """Represents an MGRS coordinate."""
    
    def __init__(self, zone_number, zone_letter, column_letter, row_letter, easting, northing, precision=5):
        self.zone_number = zone_number
        self.zone_letter = zone_letter
        self.column_letter = column_letter
        self.row_letter = row_letter
        self.easting = easting
        self.northing = northing
        self.precision = precision  # 1-5, represents digits of easting/northing
    
    def __repr__(self):
        e_str = str(int(self.easting)).zfill(self.precision)
        n_str = str(int(self.northing)).zfill(self.precision)
        return f"{self.zone_number}{self.zone_letter}{self.column_letter}{self.row_letter}{e_str}{n_str}"
    
    def to_string(self, precision=None):
        """Convert to MGRS string with specified precision (1-5)."""
        p = precision if precision else self.precision
        divisor = 10 ** (5 - p)
        e_val = int(self.easting / divisor)
        n_val = int(self.northing / divisor)
        e_str = str(e_val).zfill(p)
        n_str = str(n_val).zfill(p)
        return f"{self.zone_number}{self.zone_letter}{self.column_letter}{self.row_letter}{e_str}{n_str}"


def lat_lon_to_utm(latitude, longitude):
    """
    Convert latitude/longitude (WGS84) to UTM coordinates.
    
    Args:
        latitude: Latitude in decimal degrees (-90 to 90)
        longitude: Longitude in decimal degrees (-180 to 180)
    
    Returns:
        UTMCoordinate object
    """
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)
    
    # Calculate UTM zone number
    zone_number = int((longitude + 180) / 6) + 1
    
    # Handle special cases for Norway/Svalbard
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        zone_number = 32
    elif 72 <= latitude < 84:
        if 0 <= longitude < 9:
            zone_number = 31
        elif 9 <= longitude < 21:
            zone_number = 33
        elif 21 <= longitude < 33:
            zone_number = 35
        elif 33 <= longitude < 42:
            zone_number = 37
    
    # Calculate zone letter
    zone_letter = _get_latitude_band(latitude)
    
    # Central meridian of the zone
    lon_origin = (zone_number - 1) * 6 - 180 + 3
    lon_origin_rad = np.radians(lon_origin)
    
    # Calculate UTM coordinates
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)
    T = np.tan(lat_rad) ** 2
    C = WGS84_E2 / (1 - WGS84_E2) * np.cos(lat_rad) ** 2
    A = np.cos(lat_rad) * (lon_rad - lon_origin_rad)
    
    M = WGS84_A * (
        (1 - WGS84_E2 / 4 - 3 * WGS84_E2 ** 2 / 64 - 5 * WGS84_E2 ** 3 / 256) * lat_rad
        - (3 * WGS84_E2 / 8 + 3 * WGS84_E2 ** 2 / 32 + 45 * WGS84_E2 ** 3 / 1024) * np.sin(2 * lat_rad)
        + (15 * WGS84_E2 ** 2 / 256 + 45 * WGS84_E2 ** 3 / 1024) * np.sin(4 * lat_rad)
        - (35 * WGS84_E2 ** 3 / 3072) * np.sin(6 * lat_rad)
    )
    
    easting = UTM_K0 * N * (
        A + (1 - T + C) * A ** 3 / 6
        + (5 - 18 * T + T ** 2 + 72 * C - 58 * WGS84_E2 / (1 - WGS84_E2)) * A ** 5 / 120
    ) + 500000.0  # False easting
    
    northing = UTM_K0 * (
        M + N * np.tan(lat_rad) * (
            A ** 2 / 2
            + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
            + (61 - 58 * T + T ** 2 + 600 * C - 330 * WGS84_E2 / (1 - WGS84_E2)) * A ** 6 / 720
        )
    )
    
    # False northing for southern hemisphere
    hemisphere = 'N' if latitude >= 0 else 'S'
    if latitude < 0:
        northing += 10000000.0
    
    return UTMCoordinate(easting, northing, zone_number, zone_letter, hemisphere)


def utm_to_lat_lon(utm_coord):
    """
    Convert UTM coordinates to latitude/longitude (WGS84).
    
    Args:
        utm_coord: UTMCoordinate object or tuple (easting, northing, zone_number, zone_letter)
    
    Returns:
        Tuple (latitude, longitude) in decimal degrees
    """
    if isinstance(utm_coord, tuple):
        easting, northing, zone_number, zone_letter = utm_coord
        hemisphere = 'S' if zone_letter < 'N' else 'N'
    else:
        easting = utm_coord.easting
        northing = utm_coord.northing
        zone_number = utm_coord.zone_number
        hemisphere = utm_coord.hemisphere
    
    # Remove false easting/northing
    x = easting - 500000.0
    y = northing
    if hemisphere == 'S':
        y -= 10000000.0
    
    # Central meridian
    lon_origin = (zone_number - 1) * 6 - 180 + 3
    
    # Footpoint latitude calculation
    M = y / UTM_K0
    mu = M / (WGS84_A * (1 - WGS84_E2 / 4 - 3 * WGS84_E2 ** 2 / 64 - 5 * WGS84_E2 ** 3 / 256))
    
    e1 = (1 - np.sqrt(1 - WGS84_E2)) / (1 + np.sqrt(1 - WGS84_E2))
    
    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
        + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
        + (151 * e1 ** 3 / 96) * np.sin(6 * mu)
        + (1097 * e1 ** 4 / 512) * np.sin(8 * mu)
    )
    
    N1 = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(phi1) ** 2)
    T1 = np.tan(phi1) ** 2
    C1 = WGS84_E2 / (1 - WGS84_E2) * np.cos(phi1) ** 2
    R1 = WGS84_A * (1 - WGS84_E2) / ((1 - WGS84_E2 * np.sin(phi1) ** 2) ** 1.5)
    D = x / (N1 * UTM_K0)
    
    latitude = phi1 - (N1 * np.tan(phi1) / R1) * (
        D ** 2 / 2
        - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * WGS84_E2 / (1 - WGS84_E2)) * D ** 4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * WGS84_E2 / (1 - WGS84_E2) - 3 * C1 ** 2) * D ** 6 / 720
    )
    
    longitude = lon_origin + np.degrees(
        (D - (1 + 2 * T1 + C1) * D ** 3 / 6
         + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * WGS84_E2 / (1 - WGS84_E2) + 24 * T1 ** 2) * D ** 5 / 120)
        / np.cos(phi1)
    )
    
    return np.degrees(latitude), longitude


def utm_to_mgrs(utm_coord):
    """
    Convert UTM coordinates to MGRS.
    
    Args:
        utm_coord: UTMCoordinate object
    
    Returns:
        MGRSCoordinate object
    """
    zone_number = utm_coord.zone_number
    zone_letter = utm_coord.zone_letter
    easting = utm_coord.easting
    northing = utm_coord.northing
    
    # Calculate 100km square column letter
    # Column letters repeat every 3 zones
    set_number = (zone_number - 1) % 3
    col_index = int(easting / 100000) - 1 + (set_number * 8)
    col_index = col_index % len(MGRS_COLUMN_LETTERS)
    column_letter = MGRS_COLUMN_LETTERS[col_index]
    
    # Calculate 100km square row letter
    # Row letters repeat every 2,000,000 meters
    row_index = int(northing / 100000) % 20
    # Odd zones start at different letters
    if zone_number % 2 == 0:
        row_index = (row_index + 5) % 20
    row_letter = MGRS_ROW_LETTERS[row_index]
    
    # Get the 5-digit easting/northing within the 100km square
    grid_easting = int(easting % 100000)
    grid_northing = int(northing % 100000)
    
    return MGRSCoordinate(
        zone_number, zone_letter, column_letter, row_letter,
        grid_easting, grid_northing, precision=5
    )


def mgrs_to_utm(mgrs_coord):
    """
    Convert MGRS coordinates to UTM.
    
    Args:
        mgrs_coord: MGRSCoordinate object or MGRS string
    
    Returns:
        UTMCoordinate object
    """
    if isinstance(mgrs_coord, str):
        mgrs_coord = parse_mgrs_string(mgrs_coord)
    
    zone_number = mgrs_coord.zone_number
    zone_letter = mgrs_coord.zone_letter
    
    # Find the column letter index
    set_number = (zone_number - 1) % 3
    col_index = MGRS_COLUMN_LETTERS.index(mgrs_coord.column_letter)
    
    # Calculate base easting from column letter
    base_easting = ((col_index - set_number * 8 + 1) % 8 + 1) * 100000
    
    # Find the row letter index
    row_index = MGRS_ROW_LETTERS.index(mgrs_coord.row_letter)
    if zone_number % 2 == 0:
        row_index = (row_index - 5) % 20
    
    # Calculate base northing (need to determine the 2,000,000m block)
    # This requires knowing the latitude band
    base_northing = _get_min_northing(zone_letter)
    base_northing += row_index * 100000
    
    # Adjust for 2,000,000m repetition
    while base_northing < _get_min_northing(zone_letter):
        base_northing += 2000000
    
    # Scale easting/northing based on precision
    scale = 10 ** (5 - mgrs_coord.precision)
    grid_easting = mgrs_coord.easting * scale
    grid_northing = mgrs_coord.northing * scale
    
    easting = base_easting + grid_easting
    northing = base_northing + grid_northing
    
    hemisphere = 'S' if zone_letter < 'N' else 'N'
    
    return UTMCoordinate(easting, northing, zone_number, zone_letter, hemisphere)


def lat_lon_to_mgrs(latitude, longitude, precision=5):
    """
    Convert latitude/longitude to MGRS.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        precision: Precision level (1-5), default 5 (1m precision)
    
    Returns:
        MGRSCoordinate object
    """
    utm = lat_lon_to_utm(latitude, longitude)
    mgrs = utm_to_mgrs(utm)
    mgrs.precision = precision
    return mgrs


def mgrs_to_lat_lon(mgrs_coord):
    """
    Convert MGRS to latitude/longitude.
    
    Args:
        mgrs_coord: MGRSCoordinate object or MGRS string
    
    Returns:
        Tuple (latitude, longitude) in decimal degrees
    """
    utm = mgrs_to_utm(mgrs_coord)
    return utm_to_lat_lon(utm)


def parse_mgrs_string(mgrs_string):
    """
    Parse an MGRS string into an MGRSCoordinate object.
    
    Args:
        mgrs_string: MGRS string (e.g., "33TWN8012067890")
    
    Returns:
        MGRSCoordinate object
    """
    # Remove spaces
    mgrs_string = mgrs_string.replace(" ", "").upper()
    
    # Parse components
    pattern = r'^(\d{1,2})([C-X])([A-Z])([A-Z])(\d+)$'
    match = re.match(pattern, mgrs_string)
    
    if not match:
        raise ValueError(f"Invalid MGRS string: {mgrs_string}")
    
    zone_number = int(match.group(1))
    zone_letter = match.group(2)
    column_letter = match.group(3)
    row_letter = match.group(4)
    coords = match.group(5)
    
    # Split coordinates in half
    if len(coords) % 2 != 0:
        raise ValueError(f"Invalid MGRS coordinate length: {coords}")
    
    precision = len(coords) // 2
    easting = int(coords[:precision])
    northing = int(coords[precision:])
    
    # Scale to full 5-digit precision
    scale = 10 ** (5 - precision)
    easting *= scale
    northing *= scale
    
    return MGRSCoordinate(zone_number, zone_letter, column_letter, row_letter, 
                          easting, northing, precision)


def _get_latitude_band(latitude):
    """Get the MGRS latitude band letter for a given latitude."""
    if latitude < -80:
        return 'C'
    elif latitude >= 84:
        return 'X'
    
    band_index = int((latitude + 80) / 8)
    band_index = min(band_index, len(MGRS_LATITUDE_BANDS) - 1)
    return MGRS_LATITUDE_BANDS[band_index]


def _get_min_northing(zone_letter):
    """Get the minimum northing value for a latitude band."""
    # Approximate northing at the bottom of each band
    band_index = MGRS_LATITUDE_BANDS.index(zone_letter)
    min_lat = -80 + band_index * 8
    
    # Simple approximation (not exact, but good enough for most purposes)
    if min_lat < 0:
        return 10000000 + min_lat * 110540  # Southern hemisphere
    else:
        return min_lat * 110540


# ============================================================
# LOCAL COORDINATE TRANSFORMATIONS (For swarm simulation)
# ============================================================

class LocalCoordinateFrame:
    """
    Local tangent plane coordinate frame (ENU - East-North-Up).
    Used for swarm operations where drones work relative to a reference point.
    """
    
    def __init__(self, origin_lat, origin_lon, origin_alt=0.0):
        """
        Initialize local frame with an origin point.
        
        Args:
            origin_lat: Origin latitude in decimal degrees
            origin_lon: Origin longitude in decimal degrees
            origin_alt: Origin altitude in meters (default 0)
        """
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.origin_alt = origin_alt
        
        # Pre-calculate conversion factors at origin
        lat_rad = np.radians(origin_lat)
        self.meters_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
        self.meters_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
        
        # Store UTM origin for reference
        self.origin_utm = lat_lon_to_utm(origin_lat, origin_lon)
    
    def global_to_local(self, latitude, longitude, altitude=0.0):
        """
        Convert global coordinates to local ENU coordinates.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            altitude: Altitude in meters
        
        Returns:
            numpy array [east, north, up] in meters
        """
        d_lat = latitude - self.origin_lat
        d_lon = longitude - self.origin_lon
        d_alt = altitude - self.origin_alt
        
        east = d_lon * self.meters_per_deg_lon
        north = d_lat * self.meters_per_deg_lat
        up = d_alt
        
        return np.array([east, north, up])
    
    def local_to_global(self, east, north, up=0.0):
        """
        Convert local ENU coordinates to global coordinates.
        
        Args:
            east: East offset in meters
            north: North offset in meters
            up: Up offset in meters
        
        Returns:
            Tuple (latitude, longitude, altitude)
        """
        d_lat = north / self.meters_per_deg_lat
        d_lon = east / self.meters_per_deg_lon
        
        latitude = self.origin_lat + d_lat
        longitude = self.origin_lon + d_lon
        altitude = self.origin_alt + up
        
        return latitude, longitude, altitude
    
    def local_to_mgrs(self, east, north, up=0.0, precision=5):
        """
        Convert local coordinates to MGRS.
        
        Returns:
            MGRSCoordinate object
        """
        lat, lon, _ = self.local_to_global(east, north, up)
        return lat_lon_to_mgrs(lat, lon, precision)


# Convenience function for quick distance calculation
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: First point in decimal degrees
        lat2, lon2: Second point in decimal degrees
    
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c
