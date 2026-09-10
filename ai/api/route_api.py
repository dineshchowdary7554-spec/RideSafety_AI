# ============================================================
# RIDEGUARDIAN ROUTE MAP MATCHING API
# ============================================================

import math
from typing import List

import requests

from fastapi import APIRouter
from pydantic import BaseModel


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    tags=["Route Map Matching"]
)


# ============================================================
# REQUEST MODELS
# ============================================================

class RoutePoint(BaseModel):
    latitude: float
    longitude: float


class RouteMatchRequest(BaseModel):
    points: List[RoutePoint]


# ============================================================
# CALCULATE DISTANCE BETWEEN TWO GPS POINTS
# ============================================================

def route_distance_meters(
    point1: RoutePoint,
    point2: RoutePoint
) -> float:

    earth_radius = 6371000.0

    latitude1 = math.radians(point1.latitude)
    longitude1 = math.radians(point1.longitude)

    latitude2 = math.radians(point2.latitude)
    longitude2 = math.radians(point2.longitude)

    latitude_delta = latitude2 - latitude1
    longitude_delta = longitude2 - longitude1

    a = (
        math.sin(latitude_delta / 2) ** 2
        +
        math.cos(latitude1)
        *
        math.cos(latitude2)
        *
        math.sin(longitude_delta / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return float(
        2
        *
        earth_radius
        *
        math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )
    )


# ============================================================
# REMOVE INVALID / NEARLY DUPLICATE GPS POINTS
# ============================================================

def prepare_route_points(
    points: List[RoutePoint]
) -> List[RoutePoint]:

    if len(points) <= 2:
        return points

    cleaned_points = [
        points[0]
    ]

    for point in points[1:]:

        previous_point = cleaned_points[-1]

        distance = route_distance_meters(
            previous_point,
            point
        )

        # Ignore duplicate / extremely small GPS movement.
        if distance >= 5:
            cleaned_points.append(
                point
            )

    # Always preserve the final point.
    final_point = points[-1]
    last_point = cleaned_points[-1]

    if (
        last_point.latitude != final_point.latitude
        or
        last_point.longitude != final_point.longitude
    ):
        cleaned_points.append(
            final_point
        )

    return cleaned_points


# ============================================================
# REMOVE OBVIOUS GPS JUMPS
# ============================================================

def remove_gps_jumps(
    points: List[RoutePoint]
) -> List[RoutePoint]:

    if len(points) <= 2:
        return points

    filtered_points = [
        points[0]
    ]

    for point in points[1:]:

        previous_point = filtered_points[-1]

        distance = route_distance_meters(
            previous_point,
            point
        )

        # A single reading jumping more than 2 km
        # is treated as an obvious GPS error.
        if distance <= 2000:
            filtered_points.append(
                point
            )

    if len(filtered_points) < 2:
        return points

    return filtered_points


# ============================================================
# SELECT IMPORTANT ROUTE POINTS
# ============================================================

def select_route_points(
    points: List[RoutePoint],
    maximum_points: int = 50
) -> List[RoutePoint]:

    if len(points) <= maximum_points:
        return points

    selected_points = [
        points[0]
    ]

    step = (
        (len(points) - 1)
        /
        (maximum_points - 1)
    )

    previous_index = 0

    for index in range(
        1,
        maximum_points - 1
    ):

        point_index = round(
            index * step
        )

        point_index = max(
            previous_index + 1,
            point_index
        )

        point_index = min(
            point_index,
            len(points) - 2
        )

        selected_points.append(
            points[point_index]
        )

        previous_index = point_index

    selected_points.append(
        points[-1]
    )

    return selected_points


# ============================================================
# CREATE OSRM COORDINATE STRING
#
# OSRM:
# longitude,latitude;longitude,latitude
# ============================================================

def create_osrm_coordinate_string(
    points: List[RoutePoint]
) -> str:

    return ";".join(
        f"{point.longitude},{point.latitude}"
        for point in points
    )


# ============================================================
# REMOVE CONSECUTIVE DUPLICATE COORDINATES
# ============================================================

def remove_duplicate_coordinates(
    coordinates: list
) -> list:

    cleaned_coordinates = []

    for coordinate in coordinates:

        if len(coordinate) < 2:
            continue

        longitude = float(
            coordinate[0]
        )

        latitude = float(
            coordinate[1]
        )

        new_point = {
            "latitude": latitude,
            "longitude": longitude
        }

        if cleaned_coordinates:

            previous_point = (
                cleaned_coordinates[-1]
            )

            latitude_difference = abs(
                previous_point["latitude"]
                -
                latitude
            )

            longitude_difference = abs(
                previous_point["longitude"]
                -
                longitude
            )

            if (
                latitude_difference < 0.0000001
                and
                longitude_difference < 0.0000001
            ):
                continue

        cleaned_coordinates.append(
            new_point
        )

    return cleaned_coordinates


# ============================================================
# TRY OSRM MAP MATCHING
# ============================================================

def try_osrm_map_matching(
    points: List[RoutePoint]
):

    if len(points) < 2:
        return None

    coordinate_string = (
        create_osrm_coordinate_string(
            points
        )
    )

    osrm_url = (
        "https://router.project-osrm.org/"
        "match/v1/driving/"
        +
        coordinate_string
    )

    try:

        print(
            "Attempting OSRM map matching..."
        )

        response = requests.get(
            osrm_url,
            params={
                "overview": "full",
                "geometries": "geojson",
                "steps": "false",
                "tidy": "true",
                "gaps": "ignore",
                "radiuses": ";".join(
                    ["100"] * len(points)
                )
            },
            headers={
                "User-Agent":
                "RideGuardian-AI/1.0"
            },
            timeout=30
        )

        print(
            "OSRM MATCH STATUS:",
            response.status_code
        )

        if response.status_code != 200:

            print(
                "OSRM MATCH RESPONSE:",
                response.text[:500]
            )

            return None

        result = response.json()

        matchings = result.get(
            "matchings",
            []
        )

        if not matchings:
            return None

        all_coordinates = []

        for matching in matchings:

            geometry = matching.get(
                "geometry",
                {}
            )

            coordinates = geometry.get(
                "coordinates",
                []
            )

            all_coordinates.extend(
                coordinates
            )

        cleaned_coordinates = (
            remove_duplicate_coordinates(
                all_coordinates
            )
        )

        if len(cleaned_coordinates) >= 2:
            return cleaned_coordinates

        return None

    except Exception as error:

        print(
            "OSRM MAP MATCH ERROR:",
            str(error)
        )

        return None


# ============================================================
# TRY OSRM ROUTE SNAP
#
# Fallback when map matching is unavailable.
# ============================================================

def try_osrm_route_snap(
    points: List[RoutePoint]
):

    if len(points) < 2:
        return None

    coordinate_string = (
        create_osrm_coordinate_string(
            points
        )
    )

    osrm_url = (
        "https://router.project-osrm.org/"
        "route/v1/driving/"
        +
        coordinate_string
    )

    try:

        print(
            "Attempting OSRM route snapping..."
        )

        response = requests.get(
            osrm_url,
            params={
                "overview": "full",
                "geometries": "geojson",
                "steps": "false",
                "continue_straight": "false"
            },
            headers={
                "User-Agent":
                "RideGuardian-AI/1.0"
            },
            timeout=30
        )

        print(
            "OSRM ROUTE STATUS:",
            response.status_code
        )

        if response.status_code != 200:

            print(
                "OSRM ROUTE RESPONSE:",
                response.text[:500]
            )

            return None

        result = response.json()

        routes = result.get(
            "routes",
            []
        )

        if not routes:
            return None

        geometry = routes[0].get(
            "geometry",
            {}
        )

        coordinates = geometry.get(
            "coordinates",
            []
        )

        cleaned_coordinates = (
            remove_duplicate_coordinates(
                coordinates
            )
        )

        if len(cleaned_coordinates) >= 2:
            return cleaned_coordinates

        return None

    except Exception as error:

        print(
            "OSRM ROUTE SNAP ERROR:",
            str(error)
        )

        return None


# ============================================================
# ROUTE MATCHING ENDPOINT
# ============================================================

@router.post("/match-route")
def match_route(
    request: RouteMatchRequest
):

    original_points = request.points

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(original_points) < 2:

        return {
            "success": False,
            "matched": False,
            "matchingMethod": "NONE",
            "coordinates": [],
            "originalPointCount":
                len(original_points),
            "submittedPointCount": 0,
            "matchedPointCount": 0,
            "message":
                "At least two GPS points are required."
        }

    try:

        print(
            "========================================"
        )

        print(
            "ROUTE PROCESSING STARTED"
        )

        print(
            "ORIGINAL GPS POINTS:",
            len(original_points)
        )

        # ====================================================
        # STEP 1: CLEAN DUPLICATES
        # ====================================================

        prepared_points = (
            prepare_route_points(
                original_points
            )
        )

        # ====================================================
        # STEP 2: REMOVE GPS JUMPS
        # ====================================================

        filtered_points = (
            remove_gps_jumps(
                prepared_points
            )
        )

        # ====================================================
        # STEP 3: REDUCE EXTERNAL ROUTING INPUT
        #
        # Original ride GPS data remains untouched.
        # ====================================================

        submitted_points = (
            select_route_points(
                filtered_points,
                maximum_points=50
            )
        )

        print(
            "SUBMITTED ROUTE POINTS:",
            len(submitted_points)
        )

        if len(submitted_points) < 2:

            raise ValueError(
                "Not enough valid route points."
            )

        # ====================================================
        # STEP 4: TRUE MAP MATCHING
        # ====================================================

        matched_coordinates = (
            try_osrm_map_matching(
                submitted_points
            )
        )

        if matched_coordinates:

            print(
                "MAP MATCHING SUCCESSFUL"
            )

            print(
                "MATCHED POINTS:",
                len(matched_coordinates)
            )

            return {
                "success": True,
                "matched": True,
                "matchingMethod":
                    "OSRM_MAP_MATCH",
                "coordinates":
                    matched_coordinates,
                "originalPointCount":
                    len(original_points),
                "submittedPointCount":
                    len(submitted_points),
                "matchedPointCount":
                    len(matched_coordinates),
                "message":
                    "Route successfully matched to roads."
            }

        # ====================================================
        # STEP 5: ROUTE SNAP FALLBACK
        # ====================================================

        print(
            "MAP MATCH FAILED. "
            "TRYING ROUTE SNAP..."
        )

        routed_coordinates = (
            try_osrm_route_snap(
                submitted_points
            )
        )

        if routed_coordinates:

            print(
                "ROUTE SNAP SUCCESSFUL"
            )

            print(
                "ROUTED POINTS:",
                len(routed_coordinates)
            )

            return {
                "success": True,
                "matched": True,
                "matchingMethod":
                    "OSRM_ROUTE_SNAP",
                "coordinates":
                    routed_coordinates,
                "originalPointCount":
                    len(original_points),
                "submittedPointCount":
                    len(submitted_points),
                "matchedPointCount":
                    len(routed_coordinates),
                "message":
                    "Route was snapped to nearby roads."
            }

        # ====================================================
        # STEP 6: ORIGINAL GPS FALLBACK
        # ====================================================

        print(
            "ALL MAP SERVICES FAILED."
        )

        return {
            "success": True,
            "matched": False,
            "matchingMethod":
                "ORIGINAL_GPS",
            "coordinates": [
                {
                    "latitude":
                        point.latitude,
                    "longitude":
                        point.longitude
                }
                for point in original_points
            ],
            "originalPointCount":
                len(original_points),
            "submittedPointCount":
                len(submitted_points),
            "matchedPointCount":
                len(original_points),
            "message":
                "Road matching unavailable. "
                "Displaying the original GPS route."
        }

    except Exception as error:

        print(
            "========================================"
        )

        print(
            "ROUTE PROCESSING ERROR:",
            str(error)
        )

        print(
            "========================================"
        )

        return {
            "success": False,
            "matched": False,
            "matchingMethod":
                "ORIGINAL_GPS",
            "coordinates": [
                {
                    "latitude":
                        point.latitude,
                    "longitude":
                        point.longitude
                }
                for point in original_points
            ],
            "originalPointCount":
                len(original_points),
            "submittedPointCount": 0,
            "matchedPointCount":
                len(original_points),
            "message":
                "Route processing failed: "
                +
                str(error)
        }