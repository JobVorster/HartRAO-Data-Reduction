import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as pyfits
from scipy.optimize import curve_fit
import os
import glob
from astropy.time import Time
'''' This is the file that will be used for all the functions used in the 
radio pipeline  '''
def point_cor(filename):
	hdulist = pyfits.open(filename)
	hzperk1 = hdulist[2].header["HZPERK1"]
	errhzperk1 = hdulist[2].header["HZKERR1"]
	hzperk2 = hdulist[2].header["HZPERK2"]
	errhzperk2 = hdulist[2].header["HZKERR2"]
	NcountsLCP = hdulist[3].data['Count1']
	NcountsRCP = hdulist[3].data['Count2']
	CcountsLCP = hdulist[4].data['Count1']
	CcountsRCP = hdulist[4].data['Count2']
	ScountsLCP = hdulist[5].data['Count1']
	ScountsRCP = hdulist[5].data['Count2']
	scandist = hdulist[0].header['SCANDIST']
	hpbw = hdulist[1].header['HPBW']   
	hfnbw = hdulist[1].header['FNBW']/2.0     
	centerfreq = hdulist[2].header["CENTFREQ"]
	ottCoeff = [4.484,-0.603,-0.0280]
	offset = np.linspace(-scandist/2.0, scandist/2.0, len(hdulist[3].data['MJD']))
	LCP = np.array([NcountsLCP,CcountsLCP,ScountsLCP])
	RCP = np.array([NcountsRCP,CcountsRCP,ScountsRCP])
	LCP = LCP / hzperk1
	RCP = RCP / hzperk2
	all_peaks = np.zeros((2,3))
	all_offsets = np.zeros((2,3))
	i = 0
	pc = np.zeros(2)
	pss = np.zeros(2)
	for count_arr in [LCP,RCP]:
		for j in range(0,3):
			counts = count_arr[j]
			all_peaks[i,j],all_offsets[i,j] = baseline_fit(offset,counts,hpbw,hfnbw)
		pc[i] = getpc_coeff(all_peaks[i,:],hpbw)
		all_peaks[i,:] = pc[i]*all_peaks[i,:]
		pss[i] = calc_pss(centerfreq,all_peaks[i,1]*pc[i],ottCoeff)
		i+=1
	plt.plot(offset,LCP[1]*pss[0])
	plt.show()
	return pc[0],pc[1],pss[0],pss[1]

def get_spectra(filename):
	cwd = os.getcwd()
	cwd = cwd + filename
	os.chdir(cwd)
	filelist = glob.glob('*.fits')
	if len(filelist) == 8:
		#all_spectra is a array of the different spectra sorted:
		#Position 0 & 1 is the full integration time spectra.
		#Postion 2 & 3 is the ON pointing spectra.
		#Postion 4 - 7 is the HPN, HPS, HPE, HPW respectively.
		all_spectra = sort_files_8(filelist)
		for i in [0,2,4,6]:
			all_spectra[i]['RCP'],all_spectra[i]['LCP'],all_spectra[i]['Vlsr'],all_spectra[i+1]['RCP'],all_spectra[i+1]['LCP'],all_spectra[i+1]['Vlsr'] = total_flatten_baseline(all_spectra[i],all_spectra[i+1])
		center_pc = ave_spec(all_spectra[2],all_spectra[3])
		center_pc = ave_pols(center_pc)
		north_pc = ave_pols(all_spectra[4])
		south_pc = ave_pols(all_spectra[5])
		east_pc = ave_pols(all_spectra[6])
		west_pc = ave_pols(all_spectra[7])
		pc = get_spectra_pc(north_pc,south_pc,east_pc,west_pc,center_pc)
		fin_spectra = ave_spec(all_spectra[0],all_spectra[1])
		fin_spectra['RCP'] = fin_spectra['RCP'] * pc
		fin_spectra['LCP'] = fin_spectra['LCP'] * pc
		return fin_spectra
	elif len(filelist) == 10:
		#dictlist is a array of the different spectra sorted:
		#Position 0 - 3 is the full integration time spectra.
		#Postion 4 & 5 is the ON pointing spectra.
		#Postion 6 - 9 is the HPN, HPS, HPE, HPW respectively.
		all_spectra = sort_files_10(filelist)
		for i in [0,2,4,6,8]:
			all_spectra[i]['RCP'],all_spectra[i]['LCP'],all_spectra[i]['Vlsr'],all_spectra[i+1]['RCP'],all_spectra[i+1]['LCP'],all_spectra[i+1]['Vlsr'] = total_flatten_baseline(all_spectra[i],all_spectra[i+1])
		center_pc = ave_spec(all_spectra[4],all_spectra[5])
		center_pc = ave_pols(center_pc)
		north_pc = ave_pols(all_spectra[6])
		south_pc = ave_pols(all_spectra[7])
		east_pc = ave_pols(all_spectra[8])
		west_pc = ave_pols(all_spectra[9])
		pc = get_spectra_pc(north_pc,south_pc,east_pc,west_pc,center_pc)
		fin_spectra1 = ave_spec(all_spectra[0],all_spectra[1])
		fin_spectra2 = ave_spec(all_spectra[2],all_spectra[3])
		fin_spectra = ave_spec(fin_spectra1,fin_spectra2)
		fin_spectra['RCP'] = fin_spectra['RCP'] * pc
		fin_spectra['LCP'] = fin_spectra['LCP'] * pc
		return fin_spectra
	else:
		print("Please check your coding, there is neither 8 or 10 FITS files in the folder.")
		exit()
	
def baseline_fit(offset,counts,hpbw,hfnbw):
	p0 = [np.max(counts)-np.min(counts),offset[np.argmax(counts)],0.1,np.mean(counts),hpbw]
	coeff,var_matrix = fit_gauss_lin(offset,counts,p0)
	main_beam = np.where(np.logical_and(offset >= coeff[1]-hfnbw, offset <= coeff[1]+hfnbw))[0]
	mask = np.zeros(len(offset))
	mask[main_beam] = 1
	offset_masked = np.ma.array(offset,mask = mask)
	counts_masked = np.ma.array(counts,mask = mask)
	fit_first_null = np.poly1d(np.ma.polyfit(x=offset_masked, y=counts_masked,  deg=4))
	base1 = counts - fit_first_null(offset)
	p0  = [np.max(base1)-np.min(base1),offset[np.argmax(base1)],hpbw]
	coeff,var_matrix = gauss_fit(offset,base1,p0)
	peak = coeff[0]
	offset = coeff[1]
	return peak,offset

def gauss_fit_lin(x,*p):
	amp,mu,a,c,hpbw = p
	sigma = hpbw/(2*np.sqrt(2*np.log(2)))
	return amp*np.exp(-(x-mu)**2/(2.0*sigma**2)) + a*x +c

def fit_gauss_lin(offset,counts,p0):
	gauss_coeff,gauss_matrix = curve_fit(gauss_fit_lin,offset,counts,p0)
	return gauss_coeff,gauss_matrix
	
def gauss(x, *p):
	amp, mu, hpbw = p
	sigma = hpbw/(2*np.sqrt(2*np.log(2))) 
	return amp*np.exp(-(x-mu)**2/(2.*sigma**2))

def gauss_fit(offset,counts,p0):
	gauss_coeff,gauss_matrix = curve_fit(gauss,offset,counts,p0)
	return gauss_coeff,gauss_matrix

def getpc_coeff(peaks,hpbw):
	x = [-hpbw/2,0,hpbw/2]
	p0 = [peaks[1],0.0,hpbw]
	coeff,var_matrix = gauss_fit(x,peaks,p0)
	pc = np.exp(np.log(2) * (coeff[1]**2) / (hpbw/2)**2)
	return pc
	
def S_ott(nu,*coeff):
	a,b,c = coeff
	return 10**(a + b * np.log10(nu) + c * np.log10(nu)**2)
	
def calc_pss(centerfreq,peak,coeff):
	ott = S_ott(centerfreq,*coeff)
	pss = ott/(2.0*peak)
	return pss
	
def load_file(filename):
	hdulist = pyfits.open(filename)
	spectrum = {"Vlsr" : hdulist[2].data['Vlsr'],
				"RCP" : hdulist[2].data['Polstate1'],
				"LCP" : hdulist[2].data['Polstate4'],
				"pointing" : hdulist[0].header['SPPOINT'],
				'position' : hdulist[2].header['POSITION'],
				'date' : hdulist[0].header["DATE-OBS"],
				'object' : hdulist[0].header["OBJECT"],
				'centerfreq' : hdulist[2].header['CENTFREQ'],
				't_int' : hdulist[0].header['SPTIME'],
				'hpbw' : hdulist[1].header['HPBW']}
	return spectrum
	
def sort_files_8(filelist):
	dictlist = [dict() for x in range(8)]
	for file1 in filelist :
		#dictlist is a array of the different spectra sorted:
		#Position 0 & 1 is the full integration time spectra.
		#Postion 2 & 3 is the ON pointing spectra.
		#Postion 4 - 7 is the HPN, HPS, HPE, HPW respectively.
		
		spec = load_file(file1)
		if spec['pointing'] == 0:
			if dictlist[0] == {}:
				dictlist[0] = spec
			else : 
				dictlist[1] = spec
		if spec['pointing'] == 1:
			if spec['position'] == 'ON':
				if dictlist[2] == {}:
					dictlist[2] = spec
				else :
					dictlist[3] = spec
			if spec['position'] == "HPN":
				dictlist[4] = spec
			if spec['position'] == "HPS":
				dictlist[5] = spec
			if spec['position'] == "HPE":
				dictlist[6] = spec
			if spec['position'] == "HPW":
				dictlist[7] = spec
	return dictlist

def sort_files_10(filelist):
	dictlist = [dict() for x in range(8)]
	for file1 in filelist :
		#dictlist is a array of the different spectra sorted:
		#Position 0 - 3 is the full integration time spectra.
		#Postion 4 & 5 is the ON pointing spectra.
		#Postion 6 - 9 is the HPN, HPS, HPE, HPW respectively.
		
		spec = load_file(file1)
		if spec['pointing'] == 0:
			if dictlist[0] == {}:
				dictlist[0] = spec
			elif dictlist[1] == {} : 
				dictlist[1] = spec
			elif dictlist[2] == {} :
				dictlist[2] = spec
			else : 
				dictlist[3] = spec
		if spec['pointing'] == 1:
			if spec['position'] == 'ON':
				if dictlist[4] == {}:
					dictlist[4] = spec
				else :
					dictlist[5] = spec
			if spec['position'] == "HPN":
				dictlist[6] = spec
			if spec['position'] == "HPS":
				dictlist[7] = spec
			if spec['position'] == "HPE":
				dictlist[8] = spec
			if spec['position'] == "HPW":
				dictlist[9] = spec
	return dictlist
def freq_switch(count1,count2,vlsr1,vlsr2):
	switch1 = count1 - count2
	switch2 = count2 - count1 
	spec1_common = np.in1d(vlsr1,vlsr2)
	spec2_common = np.in1d(vlsr2,vlsr1)
	spec1_common = np.nonzero(spec1_common)
	spec2_common = np.nonzero(spec2_common)
	return switch1[spec1_common],switch2[spec2_common],vlsr1[spec1_common],vlsr2[spec2_common]
	
def get_line_range(vlsr,count):
	plt.tick_params(which = 'both',direction = 'in')
	plt.minorticks_on()
	maxindex = np.argmax(np.abs(count))
	plotrange = range(maxindex - int(len(vlsr)/3.5)-1,maxindex + int(len(vlsr)/3.5)-2)
	plt.plot(vlsr[plotrange],count[plotrange])
	plt.ylim(min(count)-max(count)/50,max(count)/4)
	plt.show()
	linerange = np.fromstring(input("What is the linerange you want to use?"),sep = ',')
	return linerange

def flatten(vlsr,count,linerange,order = 3):
	sig = np.where(np.logical_and(vlsr >= linerange[0],vlsr <= linerange[1]))
	mask = np.zeros(len(count))
	mask[sig] = 1
	fit_vlsr = np.ma.array(vlsr,mask = mask)
	fit_counts = np.ma.array(count,mask = mask)
	p = np.poly1d(np.ma.polyfit(fit_vlsr, fit_counts,order))
	basefit = p(vlsr)
	count = count - basefit
	return count

def total_flatten_baseline(spectra1,spectra2):
	Vlsr1 = spectra1['Vlsr']
	Vlsr2 = spectra2['Vlsr']
	RCP1 = spectra1['RCP']
	LCP1 = spectra1['LCP']
	RCP2 = spectra2['RCP']
	LCP2 = spectra2['LCP']
	RCP1,RCP2,Vlsr1,Vlsr2 = freq_switch(RCP1,RCP2,Vlsr1,Vlsr2)
	LCP1,LCP2,Vlsr1,Vlsr2 = freq_switch(LCP1,LCP2,Vlsr1,Vlsr2)
	LCP2 = np.abs(LCP2)
	linerange1 = get_line_range(Vlsr1,RCP1)
	RCP1 = flatten(Vlsr1,RCP1,linerange1)
	LCP1 = flatten(Vlsr1,LCP1,linerange1)
	RCP2 = flatten(Vlsr1,RCP2,linerange1)
	LCP2 = flatten(Vlsr2,LCP2,linerange1)
	return RCP1,LCP1,Vlsr1,RCP2,LCP2,Vlsr2
	
def ave_spec(spectra1,spectra2):
	t_int = [spectra1['t_int'],spectra2['t_int']]
	ave_lcp = np.average(np.vstack([spectra1['LCP'],spectra2['LCP']]),axis = 0,weights = t_int)
	ave_rcp = np.average(np.vstack([spectra1['RCP'],spectra2['RCP']]),axis = 0,weights = t_int)
	time1 = Time(spectra1['date'],scale = 'utc',format = 'isot')
	time2 = Time(spectra2['date'],scale = 'utc',format = 'isot')
	dt = (time2 - time1)/2
	midtime = time1 + dt
	avespectrum = spectra1
	avespectrum['RCP'] = ave_rcp
	avespectrum['LCP'] = ave_lcp
	avespectrum['date'] = midtime
	return avespectrum
	
def ave_pols(spectra1):
	spectra1['ave_pol'] = np.mean(np.vstack([spectra1['LCP'],spectra1['RCP']]),axis = 0)
	return spectra1
	
def fit_max(spectra,halfwidth = 5):
	hpbw = spectra['hpbw']
	peak_vel = spectra['Vlsr'][np.argmax(spectra['ave_pol'])]
	vel_chan = np.abs(spectra['Vlsr']-peak_vel).argmin()
	chans = range(vel_chan - halfwidth,vel_chan + halfwidth)
	p = np.poly1d(np.polyfit(spectra['Vlsr'][chans],spectra['ave_pol'][chans],3))
	fit_x = np.linspace(spectra['Vlsr'][min(chans)],spectra['Vlsr'][max(chans)],50)
	fit = p(fit_x)
	return fit.max()

def get_spectra_pc(north,south,east,west,center):
	hpbw = center['hpbw']
	hhpbw = hpbw/2.0
	nmax = fit_max(north)
	smax = fit_max(south)
	emax = fit_max(east)
	wmax = fit_max(west)
	cmax = fit_max(center)
	x = [-hhpbw,0,hhpbw]
	p0 = [cmax,0,hpbw]
	ns = [nmax,cmax,smax]
	ns_coeff,ns_matrix = gauss_fit(x,ns,p0)
	ew = [emax,cmax,wmax]
	ew_coeff,ew_matrix = gauss_fit(x,ew,p0)
	return np.exp(np.log(2)*(ns_coeff[1]**2+ew_coeff[1]**2)/hhpbw**2)
	