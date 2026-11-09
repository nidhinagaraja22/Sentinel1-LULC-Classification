"""Test the calculation pipeline using SYNTHETIC data, not model predictions."""
import csv
from pathlib import Path
from area_statistics import summarize


def main():
    classes = {0: 'Water', 1: 'Vegetation', 2: 'Built-up'}
    months = [f'2024-{month:02d}' for month in range(1, 13)]
    # Exactly 100 imaginary equal-area pixels per month.
    maps = [[0] * 20 + [1] * (60 - i) + [2] * (20 + i) for i in range(12)]
    result = summarize(months, maps, classes, pixel_area_m2=100)
    destination = Path(__file__).parent / 'results'
    destination.mkdir(exist_ok=True)
    with (destination / 'SYNTHETIC_monthly_statistics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])
    print('SYNTHETIC DEMO ONLY: no imagery downloaded and no model trained.')
    print('Analyzed area:', result['analyzed_area_m2'], 'square metres')
    print('Built-up coverage: 20% -> 31%; total change: +11 percentage points.')
    print('Saved:', destination / 'SYNTHETIC_monthly_statistics.csv')


if __name__ == '__main__':
    main()
