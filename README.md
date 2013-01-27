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
* Near plane Z value [0.1]
* Far plane Z value [100.0]
* Horizontal field of view value (in degrees) [70.0]

lights
-----
* Position vector [-5.0 5.0 0.0]
* Light color vector (0 to 1) [1.0 1.0 1.0]

geometry
-----
* Position vector [0.0 0.0 -10.0 1.0]
* Ambient color vector [0.05 0.05 0.05]
* Diffuse color vector [1.0 0.0 1.0]
* Specular color vector [1.0 1.0 1.0]
* Specular power value [16.0]
