# Masters-Thesis---Heart-valve-prosthetics-polymer-based-

# Project summary: Heart valve replacement by means of surgery, remains the most effective treatment of aortic
heart valve disease. Polymeric heart valve prostheses comprising of styrene-based thermoplastic
block copolymers show great promise for use in cardiovascular applications. The process
of injection moulding can lead to anisotropy in the microstructure of such polymers. This
anisotropy can mimic that found in the native aortic heart valve.
The microstructure of a specific thermoplastic block copolymer referred to as SEPS, with a
styrene weight fraction of 0.22, was analysed during the process of injection moulding and
post-injection moulding by means of synchrotron Small Angle X-ray Scattering. Regardless
of the annealing conditions employed during the injection moulding process, the microstructure
displayed extremely good orientation and negligible residual stresses. These characteristics
are paramount in ensuring that, the heart valve replacements exhibit extremely good
and anisotropic mechanical properties. Such characteristics also influence the hydrodynamic
properties of the heart valve prostheses.
Furthermore, the anisotropy in microstructure of SEPS attained, post-injection moulding, was
found to be adequately predicted by a polymer chain orientation model and a simple generalised
Newtonian, power law constitutive relation with a power law exponent of 0.53. # 

# Details of data acquisition and analysis: The SAXS data relevant to the analysis of the microstructure of SEPS with a styrene weight
fraction of 0.22, was collected by the Structured Materials group, using the synchrotron beam
line I22 [23]. The first experiment was a series of in-situ SAXS measurements during injection
moulding of SEPS. The energy of the X-ray beam was 17 keV, and the beam covered an area
equivalent to 22 500 μm2. The sample-detector distance was 3.4 m. A bench-scale mini-extruder
adapted to fit the synchrotron beam line was used for the in-situ experiment.
Molten samples of SEPS were injected at a speed of 2 mm s􀀀1 by a piston of diameter 0.015 m,
between 2 parallel aluminium plates. The volumetric flowrate Q of the molten SEPS was 0:35
cm3 s􀀀1. The mini-extruder was mounted on a moving stage, allowing the X-ray beam to map
the sample from the injection point to its edge. The temporal resolution of the X-ray scans
was 5 s. In another set of measurements, the extruder was kept at a fixed position, closest to
the injection point, which allowed for observation of the microstructure evolution during flow,
at a single point in the sample.
The main objective of this experiment, referred to as the z - axis scan, was to study the effects
of annealing, cooling rate and flowrate on the microstructure generated within the sample. The
injection temperature was kept constant at 453 K, by heating the aluminium mould plates.
The cooling time for all samples, at the end of the injection moulding process, was 20 minutes,
on average.
The second experiment was conducted post-injection moulding. In this case, x and y direction
X-ray scans were performed. These are termed the microfocus data. The energy and radius of
the beam were 14 keV and 13 μm, respectively. The sample-detector distance was 1.0 m. The
x -axis scans were performed across the thickness of the samples at 12, 27 and 42 mm from the
injection point. The y-axis scans were performed at a distance of 42 mm from the injection
point only. Tables 5.1 and 5.2 in Appendix A, list the thicknesses and annealing conditions for
the x -axis and y-axis scanned samples, respectively.
In addition, Figure 3.1 depicts both the position of the injection point with respect to the trileaflet
PHV, and a schematic of the injection moulded disc samples and X-ray scans performed
on them.
The SAXS data for x, y and z axis scans were converted to red-green-blue (RGB) images via the
DawnDiamond version 2.21 software package. Figure 3.2 depicts typical SAXS images observed
during the analysis.
DawnDiamond version 2.21 has pre-set processing commands as well as, basic Python and Java
interfaces. Azimuthal, , and polar, , angle intensity as well as d-spacing calculations were
performed using this software. Calibration with a standard reference material, was necessary.
However, the latest version of DawnDiamond lacks some of the processing capabilities necessary
to fully analyse the data and automate the process. As a result, the data were systematised
in Excel files and exported to Python for further analysis. The majority of the time was spent
developing algorithms in Python to analyse approximately 20 000 SAXS images for x and y
axis scans and 10 000 SAXS images for z -axis scans. The spatial resolution between x and y
axis scan images was 10 μm. The temporal resolution between z -axis scan images was 5 s.
The SAXS images required pre-processing prior to analysis. There was a lot of noise in the data
due to background intensity and intensity flashes arising from X-rays striking the aluminium
mould plates, in the case of the z -axis scan data. These flashes obscured some of the diffraction
patterns. A data cleaning procedure was implemented, as also detailed below:
• Data pertaining to the 2,  intensity and d-spacing profiles of the SAXS images were
arranged in dataframes in Python. This was performed to allow manual data cleaning,
where deemed necessary.
• The background noise in the data was removed automatically by identifying and subtracting
the minimum most possible intensity value, in each data-set.
• A set of conditional statements was devised in Python for each set of SAXS images, to remove
the majority of the noise arising from the intensity flashes. A peak finding algorithm was
also implemented for the purposes of automating this process.
• The resultant data from the pre-processing steps were fitted against polynomial curves using
the maximum capacity of the Singular Value Decomposition implementation of the Python
SciPy library. This was preferred against Gaussian curve fitting algorithms. Polynomial
regression was more sensitive in picking up any remaining intensity noise in the data.
• In spite of these efforts, some of the data were obscured to such great extent that their
recovery was not possible. #
