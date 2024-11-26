# Testing the loading of a tomography model
from gdrift import EarthModel3D, load_dataset


a = load_dataset("REVEAL")
example = EarthModel3D()

for key in a.keys():
    if key != "coordinates":
        example.add_quantity(key, a[key])
    else:
        example.set_coordinates(a[key])
