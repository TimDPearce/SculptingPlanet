# SculptingPlanet

Python program to constrain the parameters of a single planet which is 
sculpting a debris disc, by Tim D. Pearce. The model is from
Pearce et al. 2022, which builds on that of Pearce & Wyatt 2014. It 
assumes that a planet resides interior to the disc inner edge, and that 
planet has cleared debris out to the disc inner edge. If the disc is
eccentric, then the model also assumes that the eccentricity is driven by 
the eccentricity of the planet. Given the parameters of the star and the 
disc inner edge (and optionally their associated uncertainties), the 
program calculates the minimum possible mass, maximum possible semimajor 
axis and, if the disc is eccentric, the minimum possible eccentricity of 
the perturbing planet. It can also produce a plot showing the allowed 
region of parameter space that the planet can reside in.

To use the program, first download and unpack the ZIP file (press the 
green 'Code' button on GitHub, then 'Download ZIP', then unzip the file on
your computer). Then simply run SculptingPlanet.py in a terminal.

The default settings are for 49 Cet (HD 9672), and reproduces Fig. 7 of
Pearce et al. 2022. To change the parameters, change the values in the
'User Inputs' section of SculptingPlanet.py (you should not have to change
anything outside of that section).

Feel free to use this code, and if the results go into a publication,
then please cite Pearce et al. 2022. Also, let me know if you find any 
bugs or have any requests. Happy constraining!
