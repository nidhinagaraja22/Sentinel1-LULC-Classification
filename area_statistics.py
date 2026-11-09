"""Coverage statistics for twelve ALIGNED, equal-area classification maps.

Each map is a flat sequence of integer class IDs. None means missing data.
Use only pixels valid in ALL months so changes use a fixed denominator.
Geospatial alignment and equal pixel area must be verified upstream.
"""
import math


def summarize(months, maps, classes, pixel_area_m2):
    """Return monthly class percentages and signed percentage-point changes."""
    if len(months) != 12 or len(maps) != 12 or len(set(months)) != 12:
        raise ValueError('Supply twelve distinct month labels and twelve maps.')
    if not classes or not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError('Classes and a positive finite pixel area are required.')
    size = len(maps[0])
    if not size or any(len(image) != size for image in maps):
        raise ValueError('Maps must be nonempty and have the same pixel count.')
    for image in maps:
        if any(value is not None and value not in classes for value in image):
            raise ValueError('Map contains an unknown class ID.')
    valid = [i for i in range(size) if all(image[i] is not None for image in maps)]
    if not valid:
        raise ValueError('There are no pixels valid in every month.')
    rows = []
    first = {}
    previous = {}
    for month, image in zip(months, maps):
        for class_id, name in classes.items():
            count = sum(image[i] == class_id for i in valid)
            percentage = 100 * count / len(valid)
            first.setdefault(class_id, percentage)
            rows.append({
                'month': month,
                'class_id': class_id,
                'class_name': name,
                'area_m2': count * pixel_area_m2,
                'percentage': percentage,
                'monthly_change_pp': (percentage - previous[class_id]
                                      if class_id in previous else None),
                'change_since_first_pp': percentage - first[class_id],
            })
            previous[class_id] = percentage
    return {'valid_pixels': len(valid), 'total_pixels': size,
            'analyzed_area_m2': len(valid) * pixel_area_m2, 'rows': rows}
