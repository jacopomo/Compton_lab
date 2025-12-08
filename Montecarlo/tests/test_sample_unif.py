import numpy as np

from mc.geometry.surface import Disk, Rectangle
from mc.utils.math3d import unpack_stacked

def test_disk_sample_unif_and_unpack():
	center = np.array([0.0, 0.0, 0.0])
	radius = 1.0
	angle = 0.0  # test the non-rotated case for simple geometric checks
	d = Disk(center, radius, angle)
	n = 10000
	ps = d.sample_unif(n)

	px, py, pz = unpack_stacked(ps)

	# counts and shapes
	assert len(px) == n
	assert len(py) == n
	assert len(pz) == n

	# For angle == 0 the disk lies in the z = center[2] plane
	assert np.allclose(pz, center[2])

	# radial distances in the plane must be <= radius
	r = np.sqrt((px - center[0])**2 + (py - center[1])**2)
	assert np.all(r <= radius + 1e-12)

	# mean radial distance for uniform disk = 2/3 * R (approx)
	mean_r = r.mean()
	assert abs(mean_r - (2.0/3.0) * radius) < 0.02


def test_rectangle_sample_unif_and_unpack():
	center = np.array([0.0, 0.0, 0.0])
	width = 0.6
	height = 0.4
	angle = 0.0
	rect = Rectangle(center, width, height, angle)
	n = 10000
	ps = rect.sample_unif(n)

	px, py, pz = unpack_stacked(ps)
	
	assert len(px) == n
	assert len(py) == n
	assert len(pz) == n

	# For angle == 0 the rectangle is axis-aligned and centered at center
	assert np.allclose(np.mean(px), center[0], atol=0.02)
	assert np.allclose(np.mean(py), center[1], atol=0.02)
	assert np.allclose(pz, center[2])

	# bounds
	assert px.min() >= center[0] - height - 1e-12
	assert px.max() <= center[0] + height + 1e-12
	assert py.min() >= center[1] - width - 1e-12
	assert py.max() <= center[1] + width + 1e-12
