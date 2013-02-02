Usage
=====
raytracer [-w <image_width>] [-h <image_height>] [-s]
 [-c <camera_filename>] [-l <light_filename>]
 [-g <geometry_filename>] [-o <output_filename]

All values have sane defaults.
Using the 100 sphere seen is as easy as `raytracer -g scenes/100_spheres.txt`

Program input
=====
All program input is provided in text files with the following formats.
All numbers are assumed to be floating point values.
A vector is a space-delimited sequence of three numbers.
A value is a single number.

camera
-----
* Position vector [0.0 0.0 0.0]
* Forward vector [0.0 0.0 -1.0]
* Up vector [0.0 1.0 0.0]
* Z plane vector [0.1, 100.0, 10.0]
* Horizontal field of view value (in degrees) [60.0]

lights
-----
* Position vector [-5.0 5.0 0.0]
* Light color vector (0 to 1) [1.0 1.0 1.0]

geometry
-----
* Position vector [0.0 0.0 -10.0]
* Radius value [1.0]
* Ambient color vector [0.05 0.05 0.05]
* Diffuse color vector [1.0 0.0 1.0]
* Specular color vector [1.0 1.0 1.0]
* Specular power value [16.0]
