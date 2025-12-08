import numpy as np

from mc.geometry.volume import Cylinder, RectPrism
from mc.geometry.surface import Disk, Rectangle
from mc.utils.math3d import generate_random_directions

def test_cylinder_contains():
    n = 100
    disk = Disk(np.array([0,0,0]), 4.0, np.radians(0))
    cyl = Cylinder(disk, 10)

    on_disk = disk.sample_unif(n)
    assert np.all(cyl.contains(on_disk)) # all points on the face of the cylinder are in the cylinder

    randoms = np.stack((np.random.uniform(-2,2,n), np.random.uniform(-2,2,n),np.random.uniform(0,10,n)),axis=-1)
    assert np.all(cyl.contains(randoms)) # by construction they are all inside

def test_rect_prism_contains():
    n = 100
    rect = Rectangle(np.array([0,0,0]), 2.0, 2.0, np.radians(0))
    rp = RectPrism(rect, 10)

    on_disk = rect.sample_unif(n)
    assert np.all(rp.contains(on_disk)) # all points on the face of the prism are in the prism

    randoms = np.stack((np.random.uniform(-2,2,n), np.random.uniform(-2,2,n),np.random.uniform(0,10,n)),axis=-1)
    assert np.all(rp.contains(randoms)) # by construction they are all inside

def test_cylinder_exit_distance():
    n = 100
    dis = Disk(np.array([0,0,0]), 4.0, 0.0)
    cyl = Cylinder(dis, 10.0)

    initial_points = np.stack((np.random.uniform(-2,2,n), np.random.uniform(-2,2,n),np.random.uniform(0,10,n)),axis=-1)

    directions = generate_random_directions(n)
    distances, exit_base = cyl.exit_distance(initial_points, directions)
    assert np.all(distances>0)
    assert distances.shape == exit_base.shape

def test_rect_prism_exit_distance():
    n = 100
    rect = Rectangle(np.array([0,0,0]), 2.0, 2.0, 0.0)
    rp = RectPrism(rect, 10.0)

    initial_points = np.stack((np.random.uniform(-2,2,n), np.random.uniform(-2,2,n),np.random.uniform(0,10,n)),axis=-1)

    directions = generate_random_directions(n)
    distances, exit_base = rp.exit_distance(initial_points, directions)
    assert np.all(distances>0)
    assert distances.shape == exit_base.shape