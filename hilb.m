function y = hilb(x)
% HILBERT Hilbert transform.
%	HILBERT(X) is the Hilbert transform of the real part
%	of vector X.  The real part of the result is the original
%	real data; the imaginary part is the actual Hilbert
%	transform.  See also FFT and IFFT.
%	Charles R. Denham, January 7, 1988.
%	Revised by LS, 11-19-88.
%	Copyright (C) 1988 the MathWorks, Inc.
% Reference: Jon Claerbout, Introduction to
%            Geophysical Data Analysis.
%
test=2.^nextpow2(length(x));
 if test~= length(x)
  error(' input vector length must be a power of 2')
 end
 
yy = fft(real(x));
m = length(yy);
if m ~= 1
	h = [1; 2*ones(m/2,1); zeros(m-m/2-1,1)];
	yy(:) = yy(:).*h;
end
y = ifft(yy);
