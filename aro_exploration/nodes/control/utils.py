import numpy as np


def closest_point_on_line_to_circle(
    line_point1,
    line_point2,
    circle_center,
    circle_radius,
):
    """
    Calculate the point on a line segment between two points that is closest to a circle.
    Rules:
    1. If line_point2 is inside the circle, return line_point2
    2. If the line intersects the circle, return the intersection point closer to line_point2
    3. Otherwise, return the closest point on the line to the circle

    Parameters
    ----------
    line_point1, line_point2: numpy arrays [x, y] defining the line segment
    circle_center: numpy array [x, y] for the center of the circle
    circle_radius: radius of the circle

    Returns
    -------
    closest_point: numpy array [x, y] of the closest point
    distance: distance from the circle to the line (negative if line intersects circle)

    """
    # Convert inputs to numpy arrays for vector operations
    p1 = np.array(line_point1, dtype=float)
    p2 = np.array(line_point2, dtype=float)
    center = np.array(circle_center, dtype=float)

    # Check if p2 is inside the circle
    p2_to_center_dist = np.linalg.norm(p2 - center)
    if p2_to_center_dist <= circle_radius:
        return p2, p2_to_center_dist - circle_radius, True

    # Vector from p1 to p2
    line_vector = p2 - p1

    # Unit vector in the direction of the line
    line_length = np.linalg.norm(line_vector)
    if line_length == 0:
        return p1, np.linalg.norm(p1 - center) - circle_radius, False
    line_direction = line_vector / line_length

    # Vector from p1 to circle center
    p1_to_center = center - p1

    # Project p1_to_center onto the line direction
    projection_length = np.dot(p1_to_center, line_direction)

    # Calculate the closest point on the infinite line
    closest_point_on_infinite_line = p1 + projection_length * line_direction

    # Find the closest point on the line segment
    if projection_length <= 0:
        closest_point = p1
    elif projection_length >= line_length:
        closest_point = p2
    else:
        closest_point = closest_point_on_infinite_line

    # Vector from circle center to closest point
    center_to_closest = closest_point - center

    # Distance from center to closest point
    distance_to_center = np.linalg.norm(center_to_closest)

    # Distance from circle to line (negative if line intersects circle)
    distance = distance_to_center - circle_radius

    # Check for intersections
    is_intersecting = False
    intersection_points = []

    # Only check for intersections if the line segment might intersect the circle
    if distance <= 0:
        # Calculate intersection points using the quadratic formula
        a = np.sum(line_vector**2)
        b = 2 * np.sum(line_vector * (p1 - center))
        c = np.sum((p1 - center) ** 2) - circle_radius**2

        discriminant = b**2 - 4 * a * c

        if discriminant >= 0:  # There are intersections with the infinite line
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)

            # Check if intersection points are on the line segment (0 <= t <= 1)
            if 0 <= t1 <= 1:
                intersection_points.append((p1 + t1 * line_vector, t1))
            if 0 <= t2 <= 1:
                intersection_points.append((p1 + t2 * line_vector, t2))

            # If we have intersection points, sort them by distance to p2
            # and set the closest point to the intersection closer to p2
            if intersection_points:
                is_intersecting = True

                # Sort by parameter t (higher t means closer to p2)
                intersection_points.sort(key=lambda x: x[1], reverse=True)

                # Set the closest point to the intersection point closer to p2
                closest_point = intersection_points[0][0]

    return closest_point, distance
