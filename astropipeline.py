import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as pyfits
from astroutils import point_cor,get_spectra
'''' 
Some planning : 
- Do pointing correction and pss using continiuum drift scan (specify date)
	- Conversion to temperature
	- Fix baseline
	- Gaussian fit for pointing correction
	- Flux calibration
DONE
- Bandpass correction for spectral line (specify date)
	- Frequency switching 
	- Baseline fit
	
- Pointing correction for spectral line (specify date)
	- North-South 
	- East-West
	- Gaussian for different directions
	- Apply PSS to get final graph
	
TO-DO after : 
	- Specify only date of observation 
		- Scan through continiuum scans to take nearest date
	- Feed into pipeline to do multiple times for all files
	
'''
filename_cal = 'data/continuum_drift_scans/2006d292_07h19m34s_Cont_sharmila_VIRGO_A.fits'
LCP_cont_pc,RCP_cont_pc,LCP_pss,RCP_pss = point_cor(filename_cal)
filename_spectra = r'\data\G0096_67\2006d272'
spectra = get_spectra(filename_spectra)
plt.plot(spectra['Vlsr'],RCP_pss*spectra['RCP'],label = 'RCP')
plt.plot(spectra['Vlsr'],LCP_pss*spectra['LCP'],label = 'LCP')
plt.title(spectra['object']  + ' ' + str(spectra['date']))
plt.show()