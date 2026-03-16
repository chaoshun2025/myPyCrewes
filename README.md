It is about 18 years since I used crewes matlab open source toolbox to help start my career in UT Austin. 
Crewes toolbox has a lot of nice legacy codes written by Gary Margrave and his students. I have not used 
some part of the code for long. Today with the help of Claude.ai, it is not difficult to convert some of nice matlab 
codes to python implementation.

I have test several velocity model, exploding reflector model data generation and also test on vz2vrms and post-stack Kirk Mig.

The jupyter notebook files are demos which have been test.

TestThrustModelingAndKirchMig.ipynb
testAnticline.ipynb
testMarmousi.ipynb
testSyncline.ipynb
testWedgePlusAnticline.ipynb

The Thrust model results are here.

![alt text](https://github.com/chaoshun2025/myPyCrewes/blob/main/thrustModelAndImage.jpg)

The package includes the following codes:

afd_explode.py — Acoustic finite-difference exploding-reflector modelling.
afd_reflect.py — Compute normal-incidence reflectivity from a velocity model
afd_vmodel.py - makes simple polygonal velocity models.

burg.py - Burg (Maximum Entropy) amplitude spectral estimate.
burgpr.py  –  Burg prediction-error filter (unit lag).
convz.py — Convolution then truncation for non-minimum-phase wavelets.
onvm.py – causal convolution truncated to the length of the first argument

deconb_stack.py  –  Apply Burg deconvolution to a stacked seismic section.
deconf_stack.py  –  Apply frequency-domain deconvolution to a stacked section
deconw_stack.py  –  Apply Wiener deconvolution to a stacked section

DTW.py  –  Dynamic Time Warping: time-variant time-shift estimation
DTWs.py  –  Smooth Dynamic Time Warping: time-variant time-shift estimation

fd15mig.py  –  15-degree finite-difference time migration

ftrl.py – forward Fourier transform for real-valued signals
ifftrl.py – inverse Fourier transform to a real trace
gaussian_smoother.py  –  Smooth a 2-D velocity model with a Gaussian kernel

kirk_migz.py – Post-stack Kirchhoff depth migration
kirk_mig.py – Full-featured Kirchhoff time migration

match_pinv.py – Pseudo-inverse match filter design.
match.py – Least-squares match filter design
matchf.py - design a frequency-domain match filter.

Bagaini.py – piecewise-constant reference-velocity model for PSPI migration
pspi_mig.py – pre-stack PSPI depth migration
pspi_stack.py – zero-offset section depth migration via the PSPI algorithm

splitstepf_mig.py – split-step Fourier depth migration for v(z) media
ss_mig.py – pre-stack split-step depth migration

synclinemodel.py - build a velocity model representing a syncline in a
    stratigraphic sequence.
thrustmodel.py – Build a velocity model representing a thrust sheet.

vz2vt.py – convert V(x,z) depth model to V(x,t) and Vrms(x,t).
vzmod2vrmsmod.py – convert interval velocity model V(x,z) to Vrms(x,t).

ricker.py – generate a normalised Ricker (Mexican hat) wavelet
wavemin.py – minimum-phase wavelet for impulsive seismic sources
wavenorm.py – normalise a wavelet by one of three criteria.
wavez.py  –  Zero-phase wavelet with a realistic (Ricker-like) amplitude
wavez2.py  –  Zero-phase bandpass FIR wavelet.

wwow.py  –  WWOW wavelet extraction (time-domain match filter).
wwowf.py  –  WWOW wavelet extraction (frequency-domain match filter).
