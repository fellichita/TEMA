"""Exact RGBA compatibility and bounded work for the nine-slice renderer."""

import math
import struct
import zlib

import pytest

from app.ui import rounded


def rgba_rows(png):
    position, compressed = 8, bytearray()
    width = height = None
    while position < len(png):
        length = struct.unpack('!I', png[position:position + 4])[0]
        kind, data = png[position + 4:position + 8], png[position + 8:position + 8 + length]
        if kind == b'IHDR':
            width, height = struct.unpack('!II', data[:8])
        elif kind == b'IDAT':
            compressed.extend(data)
        position += length + 12
    raw = zlib.decompress(compressed)
    assert width is not None and height is not None
    stride = width * 4 + 1
    assert len(raw) == stride * height
    assert all(raw[y * stride] == 0 for y in range(height))
    return [raw[y * stride + 1:(y + 1) * stride] for y in range(height)]


def original_rgba(fill, outline, radius, size, stroke):
    # Frozen supersampling reference: preserve strict distance/stroke tests and rounding.
    inside = tuple(int(fill[i:i + 2], 16) for i in (1, 3, 5))
    edge = tuple(int(outline[i:i + 2], 16) for i in (1, 3, 5))
    rows = []
    for y in range(size):
        pixels = bytearray()
        for x in range(size):
            channels, coverage = [0, 0, 0], 0
            for sy in range(4):
                for sx in range(4):
                    px, py = x + (sx + .5) / 4, y + (sy + .5) / 4
                    dx = max(radius - px, px - (size - radius), 0)
                    dy = max(radius - py, py - (size - radius), 0)
                    distance = math.hypot(dx, dy) - radius
                    if distance <= 0:
                        coverage += 1
                        color = edge if distance > -stroke else inside
                        for i in range(3):
                            channels[i] += color[i]
            pixels.extend([round(v / coverage) if coverage else 0 for v in channels])
            pixels.append(round(255 * coverage / 16))
        rows.append(bytes(pixels))
    return rows


@pytest.mark.parametrize('size,radius,stroke', [
    (20, 7, 1), (64, 7, 1), (30, 10, 2), (96, 10, 2), (40, 14, 2), (128, 14, 2),
    (20, 4, 1), (20, 6, 1), (64, 6, 1), (7, 4, 1), (5, 9, 1),
    (1, 1, 1), (4, 1, 0), (20, 7, -1), (20, 7, 12), (64, 7, 12),
    (19, 9.5, .125), (21, 10.5, .875), (20, 7.125, 1.375), (20, -1, 1),
])
@pytest.mark.parametrize('fill,outline', [('#175DC8', '#2B496B'), ('#000000', '#FFFFFF')])
def test_rgba_is_exactly_equal_to_original_supersampling(size, radius, stroke, fill, outline):
    actual = rgba_rows(rounded.rounded_png(fill, outline, radius, size, stroke))
    assert actual == original_rgba(fill, outline, radius, size, stroke)


def test_large_solid_tile_does_not_supersample_every_interior_pixel(monkeypatch):
    calls, original = [], math.hypot

    def observed(dx, dy):
        calls.append((dx, dy))
        return original(dx, dy)

    rounded.rounded_png.cache_clear()
    monkeypatch.setattr(rounded.math, 'hypot', observed)
    rounded.rounded_png('#123456', '#789ABC', 7, 64, 1)
    assert len(calls) < 3 * 64 * 64, 'Solid interiors must not pay 16 samples per pixel'


@pytest.mark.parametrize('scale', [1, 1.5, 2])
@pytest.mark.parametrize('radius', [6, 7])
def test_larger_background_changes_only_the_flat_stretchable_center(scale, radius):
    old_size, new_size = round(20 * scale), round(64 * scale)
    border = round((radius + 1) * scale)
    options = {'radius': round(radius * scale), 'stroke': max(1, round(scale))}
    small = rgba_rows(rounded.rounded_png('#175DC8', '#2B496B', size=old_size, **options))
    large = rgba_rows(rounded.rounded_png('#175DC8', '#2B496B', size=new_size, **options))
    for y in range(new_size):
        source_y = y if y < border else old_size - (new_size - y) if y >= new_size - border else border
        for x in range(new_size):
            source_x = x if x < border else old_size - (new_size - x) if x >= new_size - border else border
            assert large[y][x * 4:x * 4 + 4] == small[source_y][source_x * 4:source_x * 4 + 4]
