import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  ActivityIndicator,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

import { router, useLocalSearchParams } from "expo-router";

import {
  Camera,
  GeoJSONSource,
  Layer,
  Map,
  ViewAnnotation,
} from "@maplibre/maplibre-react-native";

import {
  getRideReadings,
  getRides,
  type Ride,
  type RideReading,
} from "../services/database";

// ==========================================
// MAP STYLE
// ==========================================

const MAP_STYLE_URL =
  "https://tiles.openfreemap.org/styles/liberty";


// ============================================================
// ROUTE MATCHING BACKEND
// ============================================================

const RIDEGUARDIAN_API_URL =
  (
    process.env.EXPO_PUBLIC_RIDEGUARDIAN_API_URL ||
    "https://diplomatic-vitality-production-9ccc.up.railway.app"
  ).replace(/\/$/, "");

// ==========================================
// TYPES
// ==========================================

type RouteCoordinate = {
  latitude: number;
  longitude: number;
};

type RouteMatchResponse = {
  success: boolean;
  matched: boolean;
  matchingMethod?: string;
  coordinates?: RouteCoordinate[];
  originalPointCount?: number;
  submittedPointCount?: number;
  matchedPointCount?: number;
  message?: string;
};

// ==========================================
// COMPONENT
// ==========================================

export default function RideRouteScreen() {
  const params = useLocalSearchParams<{
    id?: string;
  }>();

  const rideId = Number(params.id);

  const [ride, setRide] =
    useState<Ride | null>(null);

  const [readings, setReadings] =
    useState<RideReading[]>([]);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  // The raw route is always preserved locally.
  // Matched coordinates are used only for map display.
  const [
    matchedRouteCoordinates,
    setMatchedRouteCoordinates,
  ] = useState<RouteCoordinate[]>([]);

  const [
    routeMatching,
    setRouteMatching,
  ] = useState(false);

  const [
    matchingMethod,
    setMatchingMethod,
  ] = useState("ORIGINAL_GPS");

  // ==========================================
  // LOAD RIDE DATA
  // ==========================================

  const loadRoute = useCallback(async () => {
    try {
      setLoading(true);
      setError("");

      if (!Number.isFinite(rideId)) {
        throw new Error("Invalid ride ID.");
      }

      const rides = await getRides();

      const selectedRide = rides.find(
        (item) => Number(item.id) === rideId
      );

      if (!selectedRide) {
        throw new Error("Ride not found.");
      }

      const savedReadings =
        await getRideReadings(rideId);

      setRide(selectedRide);
      setReadings(savedReadings);
    } catch (e) {
      console.error(
        "RIDE ROUTE ERROR:",
        e
      );

      setError(
        e instanceof Error
          ? e.message
          : "Unable to load this ride route."
      );
    } finally {
      setLoading(false);
    }
  }, [rideId]);

  // ==========================================
  // INITIAL LOAD
  // ==========================================

  useEffect(() => {
    loadRoute();
  }, [loadRoute]);

  // ==========================================
  // CONVERT SQLITE GPS DATA
  // ==========================================

  const routeCoordinates =
    useMemo<RouteCoordinate[]>(() => {
      // Native background readings and foreground readings can
      // reach SQLite in a different insertion order. Always sort
      // by the actual reading timestamp before drawing the route.
      const getTimestamp = (reading: RideReading): number => {
        const rawTimestamp = (reading as any).timestamp;

        if (typeof rawTimestamp === "number") {
          return rawTimestamp;
        }

        const numericTimestamp = Number(rawTimestamp);

        if (Number.isFinite(numericTimestamp)) {
          return numericTimestamp;
        }

        const parsedTimestamp =
          new Date(String(rawTimestamp)).getTime();

        return Number.isFinite(parsedTimestamp)
          ? parsedTimestamp
          : 0;
      };

      const sortedReadings = [...readings].sort(
        (a, b) =>
          getTimestamp(a) - getTimestamp(b)
      );

      const points: RouteCoordinate[] = [];

      for (const reading of sortedReadings) {
        const latitude =
          Number(reading.latitude);

        const longitude =
          Number(reading.longitude);

        // Ignore invalid or empty GPS coordinates.
        if (
          !Number.isFinite(latitude) ||
          !Number.isFinite(longitude) ||
          Math.abs(latitude) > 90 ||
          Math.abs(longitude) > 180 ||
          (latitude === 0 && longitude === 0)
        ) {
          continue;
        }

        const previousPoint =
          points[points.length - 1];

        // Remove consecutive duplicate coordinates so the route
        // does not contain unnecessary zero-length segments.
        if (
          previousPoint &&
          Math.abs(
            previousPoint.latitude - latitude
          ) < 0.000001 &&
          Math.abs(
            previousPoint.longitude - longitude
          ) < 0.000001
        ) {
          continue;
        }

        points.push({
          latitude,
          longitude,
        });
      }

      return points;
    }, [readings]);

  // ==========================================
  // SELECT NECESSARY GPS POINTS
  //
  // This reduces network traffic while preserving
  // the beginning, end and route progression.
  // The original SQLite readings are never changed.
  // ==========================================

  const selectPointsForMatching = useCallback(
    (
      points: RouteCoordinate[],
      maximumPoints = 50
    ): RouteCoordinate[] => {
      if (points.length <= maximumPoints) {
        return points;
      }

      const selected: RouteCoordinate[] = [
        points[0],
      ];

      const step =
        (points.length - 1) /
        (maximumPoints - 1);

      let previousIndex = 0;

      for (
        let index = 1;
        index < maximumPoints - 1;
        index += 1
      ) {
        let pointIndex =
          Math.round(index * step);

        pointIndex = Math.max(
          previousIndex + 1,
          pointIndex
        );

        pointIndex = Math.min(
          pointIndex,
          points.length - 2
        );

        selected.push(
          points[pointIndex]
        );

        previousIndex = pointIndex;
      }

      selected.push(
        points[points.length - 1]
      );

      return selected;
    },
    []
  );

  // ==========================================
  // MATCH ROUTE WITH BACKEND
  //
  // The screen loads the local ride immediately.
  // Road matching then happens in the background.
  // If anything fails, the original GPS route
  // continues to be displayed.
  // ==========================================

  useEffect(() => {
    let cancelled = false;

    const matchRoute = async () => {
      setMatchedRouteCoordinates([]);
      setMatchingMethod("ORIGINAL_GPS");

      if (routeCoordinates.length < 2) {
        return;
      }

      if (!RIDEGUARDIAN_API_URL) {
        console.log(
          "ROUTE MATCHING: Backend URL not configured."
        );
        return;
      }

      const pointsToMatch =
        selectPointsForMatching(
          routeCoordinates,
          50
        );

      if (pointsToMatch.length < 2) {
        return;
      }

      try {
        setRouteMatching(true);

        console.log(
          "ROUTE MATCHING: Sending",
          pointsToMatch.length,
          "of",
          routeCoordinates.length,
          "GPS points."
        );

        const response = await fetch(
          `${RIDEGUARDIAN_API_URL}/match-route`,
          {
            method: "POST",
            headers: {
              "Content-Type":
                "application/json",
              Accept:
                "application/json",
            },
            body: JSON.stringify({
              points: pointsToMatch,
            }),
          }
        );

        if (!response.ok) {
          throw new Error(
            `Route API returned ${response.status}.`
          );
        }

        const result =
          (await response.json()) as
            RouteMatchResponse;

        const validMatchedPoints =
          Array.isArray(result.coordinates)
            ? result.coordinates
                .map((point) => ({
                  latitude: Number(
                    point.latitude
                  ),
                  longitude: Number(
                    point.longitude
                  ),
                }))
                .filter(
                  (point) =>
                    Number.isFinite(
                      point.latitude
                    ) &&
                    Number.isFinite(
                      point.longitude
                    ) &&
                    Math.abs(
                      point.latitude
                    ) <= 90 &&
                    Math.abs(
                      point.longitude
                    ) <= 180
                )
            : [];

        if (
          !cancelled &&
          result.success &&
          result.matched &&
          validMatchedPoints.length >= 2
        ) {
          setMatchedRouteCoordinates(
            validMatchedPoints
          );

          setMatchingMethod(
            result.matchingMethod ||
              "ROAD_MATCHED"
          );

          console.log(
            "ROUTE MATCHING SUCCESS:",
            result.matchingMethod,
            validMatchedPoints.length,
            "display points."
          );
        } else if (!cancelled) {
          setMatchedRouteCoordinates([]);
          setMatchingMethod(
            result.matchingMethod ||
              "ORIGINAL_GPS"
          );

          console.log(
            "ROUTE MATCHING FALLBACK:",
            result.message ||
              "Using original GPS route."
          );
        }
      } catch (matchingError) {
        if (!cancelled) {
          setMatchedRouteCoordinates([]);
          setMatchingMethod(
            "ORIGINAL_GPS"
          );

          console.warn(
            "ROUTE MATCHING ERROR:",
            matchingError
          );
        }
      } finally {
        if (!cancelled) {
          setRouteMatching(false);
        }
      }
    };

    matchRoute();

    return () => {
      cancelled = true;
    };
  }, [
    routeCoordinates,
    selectPointsForMatching,
  ]);

  // ==========================================
  // DISPLAY ROUTE
  //
  // Prefer the backend road-matched route.
  // Always fall back to the original local GPS.
  // ==========================================

  const displayRouteCoordinates =
    useMemo<RouteCoordinate[]>(() => {
      if (
        matchedRouteCoordinates.length >= 2
      ) {
        return matchedRouteCoordinates;
      }

      return routeCoordinates;
    }, [
      matchedRouteCoordinates,
      routeCoordinates,
    ]);

  // ==========================================
  // GEOJSON ROUTE
  // ==========================================

  const routeGeoJSON = useMemo(() => {
    return {
      type: "Feature" as const,

      properties: {},

      geometry: {
        type: "LineString" as const,

        coordinates:
          displayRouteCoordinates.map(
            (point) => [
              point.longitude,
              point.latitude,
            ]
          ),
      },
    };
  }, [displayRouteCoordinates]);

  // ==========================================
  // START COORDINATE
  // ==========================================

  const startCoordinate = useMemo(() => {
    if (displayRouteCoordinates.length === 0) {
      return null;
    }

    return displayRouteCoordinates[0];
  }, [displayRouteCoordinates]);

  // ==========================================
  // END COORDINATE
  // ==========================================

  const endCoordinate = useMemo(() => {
    if (displayRouteCoordinates.length === 0) {
      return null;
    }

    return displayRouteCoordinates[
      displayRouteCoordinates.length - 1
    ];
  }, [displayRouteCoordinates]);

  // ==========================================
  // CALCULATE MAP BOUNDS
  // ==========================================

  const routeBounds = useMemo(() => {
    if (displayRouteCoordinates.length === 0) {
      return [
        78.4367,
        17.335,
        78.5367,
        17.435,
      ] as [
        number,
        number,
        number,
        number
      ];
    }

    const latitudes =
      displayRouteCoordinates.map(
        (point) => point.latitude
      );

    const longitudes =
      displayRouteCoordinates.map(
        (point) => point.longitude
      );

    let minLatitude =
      Math.min(...latitudes);

    let maxLatitude =
      Math.max(...latitudes);

    let minLongitude =
      Math.min(...longitudes);

    let maxLongitude =
      Math.max(...longitudes);

    // Add padding if all points are
    // extremely close together

    if (
      minLatitude === maxLatitude
    ) {
      minLatitude -= 0.005;
      maxLatitude += 0.005;
    }

    if (
      minLongitude === maxLongitude
    ) {
      minLongitude -= 0.005;
      maxLongitude += 0.005;
    }

    return [
      minLongitude,
      minLatitude,
      maxLongitude,
      maxLatitude,
    ] as [
      number,
      number,
      number,
      number
    ];
  }, [displayRouteCoordinates]);

  // ==========================================
  // LOADING SCREEN
  // ==========================================

  if (loading) {
    return (
      <SafeAreaView
        style={styles.container}
      >
        <View style={styles.center}>
          <ActivityIndicator size="large" />

          <Text
            style={styles.loadingText}
          >
            Loading route...
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  // ==========================================
  // ERROR SCREEN
  // ==========================================

  if (error || !ride) {
    return (
      <SafeAreaView
        style={styles.container}
      >
        <View style={styles.center}>
          <Text
            style={styles.errorTitle}
          >
            Unable to load route
          </Text>

          <Text
            style={styles.errorText}
          >
            {error || "Ride not found."}
          </Text>

          <TouchableOpacity
            style={
              styles.backButtonLarge
            }
            onPress={() =>
              router.back()
            }
          >
            <Text
              style={styles.backButtonText}
            >
              GO BACK
            </Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // ==========================================
  // MAIN SCREEN
  // ==========================================

  return (
    <SafeAreaView
      style={styles.container}
    >
      {/* HEADER */}

      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() =>
            router.back()
          }
        >
          <Text style={styles.backText}>
            ‹
          </Text>
        </TouchableOpacity>

        <View style={styles.headerText}>
          <Text
            style={styles.smallTitle}
          >
            RIDE ROUTE
          </Text>

          <Text style={styles.title}>
            Ride #{ride.id}
          </Text>

          <Text style={styles.routeStatus}>
            {routeMatching
              ? "MATCHING ROAD ROUTE..."
              : matchingMethod ===
                  "ORIGINAL_GPS"
                ? "GPS ROUTE"
                : "ROAD MATCHED"}
          </Text>
        </View>
      </View>

      {/* MAP */}

      <View style={styles.mapContainer}>
        {displayRouteCoordinates.length >= 2 ? (
          <Map
            style={
              StyleSheet.absoluteFill
            }
            mapStyle={MAP_STYLE_URL}
          >
            {/* CAMERA */}

            <Camera
              initialViewState={{
                bounds: routeBounds,
              }}
              padding={{
                top: 70,
                right: 50,
                bottom: 120,
                left: 50,
              }}
            />

            {/* ROUTE SOURCE */}

            <GeoJSONSource
              id="ride-route-source"
              data={routeGeoJSON}
            >
              {/* ROUTE LINE */}

              <Layer
                id="ride-route-line"
                type="line"
                source="ride-route-source"
                paint={{
                  "line-color": "#18D6A3",
                  "line-width": 5,
                }}
                layout={{
                  "line-cap": "round",
                  "line-join": "round",
                }}
              />
            </GeoJSONSource>

            {/* START MARKER */}

            {startCoordinate && (
              <ViewAnnotation
                id="start-marker"
                lngLat={[
                  startCoordinate.longitude,
                  startCoordinate.latitude,
                ]}
              >
                <View
                  style={
                    styles.startMarker
                  }
                >
                  <View
                    style={
                      styles.markerInner
                    }
                  />
                </View>
              </ViewAnnotation>
            )}

            {/* END MARKER */}

            {endCoordinate && (
              <ViewAnnotation
                id="end-marker"
                lngLat={[
                  endCoordinate.longitude,
                  endCoordinate.latitude,
                ]}
              >
                <View
                  style={
                    styles.endMarker
                  }
                >
                  <View
                    style={
                      styles.markerInner
                    }
                  />
                </View>
              </ViewAnnotation>
            )}
          </Map>
        ) : (
          <View style={styles.noRoute}>
            <Text
              style={styles.noRouteTitle}
            >
              Route unavailable
            </Text>

            <Text
              style={styles.noRouteText}
            >
              This ride does not contain
              enough valid GPS points to
              draw a route.
            </Text>
          </View>
        )}
      </View>

      {/* RIDE INFORMATION */}

      <View style={styles.infoCard}>
        <View style={styles.infoItem}>
          <Text
            style={styles.infoLabel}
          >
            DISTANCE
          </Text>

          <Text
            style={styles.infoValue}
          >
            {Number(
              ride.distance || 0
            ).toFixed(2)}{" "}
            km
          </Text>
        </View>

        <View style={styles.infoItem}>
          <Text
            style={styles.infoLabel}
          >
            AVG SPEED
          </Text>

          <Text
            style={styles.infoValue}
          >
            {Number(
              ride.averageSpeed || 0
            ).toFixed(1)}{" "}
            km/h
          </Text>
        </View>

        <View style={styles.infoItem}>
          <Text
            style={styles.infoLabel}
          >
            GPS POINTS
          </Text>

          <Text
            style={styles.infoValue}
          >
            {routeCoordinates.length}
          </Text>
        </View>
      </View>

      <View style={styles.matchingInfo}>
        <Text style={styles.matchingInfoText}>
          {routeMatching
            ? "Improving route with road data..."
            : matchingMethod === "ORIGINAL_GPS"
              ? "Showing original GPS route."
              : "Showing AI-cleaned road route."}
        </Text>
      </View>
    </SafeAreaView>
  );
}

// ==========================================
// STYLES
// ==========================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#020407",
  },

  header: {
    height: 76,
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 18,
    backgroundColor: "#07152F",
  },

  backButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "#121C2D",
    borderWidth: 1,
    borderColor: "#263750",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 14,
  },

  backText: {
    color: "#FFFFFF",
    fontSize: 34,
    lineHeight: 38,
  },

  headerText: {
    flex: 1,
  },

  smallTitle: {
    color: "#18D6A3",
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 1.3,
  },

  title: {
    color: "#FFFFFF",
    fontSize: 23,
    fontWeight: "800",
    marginTop: 3,
  },

  routeStatus: {
    color: "#8192AB",
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 0.8,
    marginTop: 4,
  },

  mapContainer: {
    flex: 1,
    overflow: "hidden",
    backgroundColor: "#0D1727",
  },

  noRoute: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 30,
    backgroundColor: "#0D1727",
  },

  noRouteTitle: {
    color: "#FFFFFF",
    fontSize: 20,
    fontWeight: "800",
  },

  noRouteText: {
    color: "#8192AB",
    textAlign: "center",
    marginTop: 10,
    lineHeight: 20,
  },

  startMarker: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: "#18D6A3",
    borderWidth: 3,
    borderColor: "#FFFFFF",
    justifyContent: "center",
    alignItems: "center",
  },

  endMarker: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: "#FF4B55",
    borderWidth: 3,
    borderColor: "#FFFFFF",
    justifyContent: "center",
    alignItems: "center",
  },

  markerInner: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: "#FFFFFF",
  },

  infoCard: {
    minHeight: 82,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 18,
    backgroundColor: "#0D1727",
    borderTopWidth: 1,
    borderTopColor: "#1E304A",
  },

  infoItem: {
    flex: 1,
  },

  infoLabel: {
    color: "#64748B",
    fontSize: 8,
    fontWeight: "800",
  },

  infoValue: {
    color: "#FFFFFF",
    fontSize: 14,
    fontWeight: "800",
    marginTop: 5,
  },

  matchingInfo: {
    minHeight: 34,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 18,
    backgroundColor: "#07152F",
    borderTopWidth: 1,
    borderTopColor: "#1E304A",
  },

  matchingInfoText: {
    color: "#8192AB",
    fontSize: 10,
    fontWeight: "600",
  },

  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 30,
  },

  loadingText: {
    color: "#8192AB",
    marginTop: 12,
  },

  errorTitle: {
    color: "#FFFFFF",
    fontSize: 20,
    fontWeight: "800",
  },

  errorText: {
    color: "#8192AB",
    textAlign: "center",
    marginTop: 10,
  },

  backButtonLarge: {
    marginTop: 22,
    height: 48,
    paddingHorizontal: 24,
    borderRadius: 13,
    backgroundColor: "#168CF5",
    justifyContent: "center",
    alignItems: "center",
  },

  backButtonText: {
    color: "#FFFFFF",
    fontWeight: "800",
    fontSize: 11,
  },
});
