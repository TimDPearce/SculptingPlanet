
'''Program to constrain the parameters of a single planet which is 
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

To use the program, simply change the values in the 'User Inputs' section
just below. You should not have to change anything outside of that 
section. The default settings are for 49 Cet (HD 9672), and reproduces 
Fig. 7 of Pearce et al. 2022.

Feel free to use this code, and if the results go into a publication,
then please cite Pearce et al. 2022. Also, let me know if you find any 
bugs or have any requests. Happy constraining!'''

############################### Libraries ###############################
import numpy as np
import math
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
############################# User Inputs #############################
''' Parameters of the system, and their associated uncertainties and 
units. For example, if the star mass is 1.2 MSun with 1sigma 
uncertainties of +0.1 MSun and -0.2 MSun, then set mStar_mSun=2, 
mStar1SigUp_mSun=0.1 and mStar1SigDown_mSun=0.2. If you do not want
to consider an uncertainty, use numpy NaN, e.g. 
mStar1SigUp_mSun = np.nan. You should not need to change any part of 
the code other than that in this section.'''

# Star mass, in Solar masses
mStar_mSun = 1.98
mStar1SigUp_mSun = 0.01
mStar1SigDown_mSun = 0.01

# System age, in myr
age_myr = 45.
age1SigUp_myr = 5.
age1SigDown_myr = 5.

# Pericentre and apocentre of the disc inner edge, in au (if disc is
# axisymmetric, then these will equal each other)
discInnerEdgePeri_au = 62.5
discInnerEdgePeri1SigUp_au = 4.0
discInnerEdgePeri1SigDown_au = 4.5

discInnerEdgeApo_au = 62.5
discInnerEdgeApo1SigUp_au = 4.0
discInnerEdgeApo1SigDown_au = 4.5

# Should a plot be produced
shouldPlotBeMade = True

################# Global variables (don't change these!) ################
if discInnerEdgePeri_au != discInnerEdgeApo_au: isDiscAsymmetric = True
else: isDiscAsymmetric = False

############################ Maths functions ############################
def RoundNumberToDesiredSigFigs(num, sigFigs=None):
	'''Returns a float of a given number to the specified significant 
	figures'''

	# Default number of decimal digits (in case precision unspecified)
	defaultSigFigs = 2

	# Get sig figs if not defined
	if sigFigs is None:
		sigFigs = defaultSigFigs
			
	# Catch case if number is zero
	if num == 0:
		exponent = 0
	
	# Otherwise number is non-zero
	else:
		exponent = GetBase10OrderOfNumber(num)
	
	# Get the coefficient
	coefficient = round(num / float(10**exponent), sigFigs-1)

	roundedNumber = coefficient * 10**exponent
	
	# Get the decimal places to round to (to avoid computer rounding errors)
	decimalPlacesToRoundTo = sigFigs-exponent
	
	roundedNumber = round(roundedNumber, decimalPlacesToRoundTo)
	
	return roundedNumber
	
#------------------------------------------------------------------------
def GetBase10OrderOfNumber(number):
	'''Return the order of the positive number in base 10, e.g. inputting
	73 returns 1 (since 10^1 < 73 < 10^2).'''
	
	if number <= 0: return np.nan
	
	return int(math.floor(np.log10(abs(number))))

#------------------------------------------------------------------------
def GetGaussianAsymmetricUncertainties(dependencyUncertainties1SigUp_unit, dependencyUncertainties1SigDown_unit, differentialValueWRTDependencies_unitPerUnit):
	'''Get the ~Gaussian uncertainties on a value, where its 
	dependencies may have asymmetric errors. Takes dictionaries of the
	positive and negative untertainties on the dependencies, and the
	derivatives of the value wrt its dependencies'''
	
	# Contributions to the total squared errors from the different 
	# sources
	value1SigUpSquared_unit2, value1SigDownSquared_unit2 = 0., 0.

	for dependencyName in dependencyUncertainties1SigUp_unit:
		dependencyValue1SigUp_unit = dependencyUncertainties1SigUp_unit[dependencyName]
		dependencyValue1SigDown_unit = dependencyUncertainties1SigDown_unit[dependencyName]
		
		differentialValueWRTDependency_unitPerUnit = differentialValueWRTDependencies_unitPerUnit[dependencyName]
	
		# Catch overflow error if derivatives very large (e.g. 
		# Laplace coefficient as alpha->1)
		differentialValueWRTDependency_unitPerUnit = min(differentialValueWRTDependency_unitPerUnit, 1e99)
		differentialValueWRTDependency_unitPerUnit = max(differentialValueWRTDependency_unitPerUnit, -1e99)
			
		# If the value increases as this dependency increases, then 
		# positive value error corresponds to positive dependency error etc.
		if differentialValueWRTDependency_unitPerUnit >= 0:	
			value1SigUpSquared_unit2 += (differentialValueWRTDependency_unitPerUnit * dependencyValue1SigUp_unit)**2
			value1SigDownSquared_unit2 += (differentialValueWRTDependency_unitPerUnit * dependencyValue1SigDown_unit)**2
		
		# Otherwise the plt mass decreases as this variable 
		# increases, so positive mass error corresponds to negative 
		# variable error etc.
		else:
			value1SigUpSquared_unit2 += (differentialValueWRTDependency_unitPerUnit * dependencyValue1SigDown_unit)**2
			value1SigDownSquared_unit2 += (differentialValueWRTDependency_unitPerUnit * dependencyValue1SigUp_unit)**2

	# Finally, get the plt mass errors
	value1SigUp_unit = value1SigUpSquared_unit2**0.5
	value1SigDown_unit = value1SigDownSquared_unit2**0.5

	return value1SigUp_unit, value1SigDown_unit
	
########################### Dynamics functions ##########################
def GetPearce2022MinimumPlanetMassAndLocationAndEccentricity_mJupAndAu():
	'''Find the minimum possible plt mass that can carve out the inner
	edge of a debris disc, according to Pearce and Wyatt (2022)'''
	
	aStep_au = discInnerEdgeApo_au / 1000.0
	
	# Work from largest distance to smaller distances, so first 
	# compatible plt mass found is the minimum allowed
	as_au = np.arange(discInnerEdgeApo_au - aStep_au, 1e-99, -aStep_au)

	smallestAllowedPlanetMass_mJup = np.nan
	semimajorAxisOfSmallestAllowedPlanet_au = np.nan
	eccentricityOfSmallestAllowedPlanet = np.nan

	for a_au in as_au:

		mPltFromHillConstraint_mJup, pltEccentricityFromHillConstraint = GetPltMassAndEccentricityFromHillConstraint_mJup(a_au, mStar_mSun, discInnerEdgePeri_au, discInnerEdgeApo_au)

		mPltFromDiffusionTime_mJup = GetPltMassFromDiffusionTimeConstraint_mJup(a_au, mStar_mSun, age_myr, discInnerEdgeApo_au)
		
		mPltFromSecularTimeConstraint_mJup = GetPltMassAndAbsErrorsFromSecularTimeConstraint_mJup(a_au, mStar_mSun, age_myr, discInnerEdgeApo_au)[0]
		
		# Determine whether this is the smallest allowed plt
		if(mPltFromHillConstraint_mJup > mPltFromDiffusionTime_mJup and mPltFromHillConstraint_mJup > mPltFromSecularTimeConstraint_mJup):
			smallestAllowedPlanetMass_mJup = mPltFromHillConstraint_mJup
			
			semimajorAxisOfSmallestAllowedPlanet_au = a_au
			
			# Only update the plt eccentricity if the disc model
			# is asymmetric
			if isDiscAsymmetric:
				eccentricityOfSmallestAllowedPlanet = pltEccentricityFromHillConstraint
							
			break
	
	# Package up the parameters
	minMassPlanetPars = {'mPlt_mJup': smallestAllowedPlanetMass_mJup,
						'pltSemimajorAxis_au': semimajorAxisOfSmallestAllowedPlanet_au,
						'pltEccentricity': eccentricityOfSmallestAllowedPlanet}
	
	return minMassPlanetPars
	
#------------------------------------------------------------------------
def GetPltMassAndEccentricityFromHillConstraint_mJup(aplt_au, mStar_mSun, discInnerEdgePeri_au, discInnerEdgeApo_au):
	'''The plt must be at least 5 Hill radii from the apocentre of 
	the inner disc edge (Pearce and Wyatt 2014), and an eccentric plt
	would drive up the inner edge eccentricity'''

	# Find the plt eccentricity at this semimajor axis. Catch any
	# rounding errors that would make eccentricity negative
	if isDiscAsymmetric:
		eplt = 0.4 * (discInnerEdgeApo_au - discInnerEdgePeri_au) / aplt_au
		
		# If the semimajor axis is too small, eccentricity would have to 
		# be more than one. Unphysical, so nan
		if eplt >= 1: return np.nan, np.nan

	else:
		eplt = 0.
	
	if eplt < 0:
		print('+++ WARNING: plt eccentricity negative in Hill constraint calculation')
	
	# Catch breakdown in equation if Q_plt > discInnerEdgeApo_au
	if aplt_au * (1.+eplt) > discInnerEdgeApo_au:
		return np.nan, np.nan
		
	mPlt_mJup = 8.38 * mStar_mSun * (3.-eplt) * (discInnerEdgeApo_au / (aplt_au * (1.+eplt)) - 1.)**3
		
	return mPlt_mJup, eplt

#------------------------------------------------------------------------
def GetPltMassFromDiffusionTimeConstraint_mJup(a_au, mStar_mSun, age_myr, discInnerEdgeApo_au):
	'''More than 10 diffusion times must have ellapsed for the plt to 
	have cleared a gap (Pearce and Wyatt 2014)'''

	mPlt_mJup = 0.3313 * a_au * mStar_mSun**.75 * discInnerEdgeApo_au**-.25 * age_myr**-.5

	return mPlt_mJup
	
#------------------------------------------------------------------------
def GetPltMassAndAbsErrorsFromSecularTimeConstraint_mJup(a_au, mStar_mSun, age_myr, discInnerEdgeApo_au, mStar1SigUp_mSun=0, mStar1SigDown_mSun=0, age1SigUp_myr=0, age1SigDown_myr=0, discInnerEdgeApo1SigUp_au=0, discInnerEdgeApo1SigDown_au=0, apltError_au=0):
	'''More than 10 secular times must have ellapsed for the plt to
	 have cleared a gap (Pearce and Wyatt 2014). For the given plt 
	 semimajor axis, returns the plt mass and asymmetric errors'''
		
	alpha = a_au / discInnerEdgeApo_au

	b1_32 = GetLaplaceCoefficient(1.0, 1.5, alpha)

	mPlt_mJup = 0.04192 * age_myr**-1 * mStar_mSun**.5 * a_au**-1 * discInnerEdgeApo_au**2.5 * b1_32**-1

	# Don't calculate errors if none supplied
	if mStar1SigUp_mSun==0 and age1SigUp_myr==0 and discInnerEdgeApo1SigUp_au==0 and discInnerEdgeApo1SigDown_au==0 and apltError_au==0:
		return mPlt_mJup, 0., 0.
	
	fracErr1SigUp, fracErr1SigDown = GetFracErrorOnPltMassAtSemimajorAxisFromSecularTimeConstraint(alpha, b1_32, mStar_mSun, mStar1SigUp_mSun, mStar1SigDown_mSun, age_myr, age1SigUp_myr, age1SigDown_myr, discInnerEdgeApo_au, discInnerEdgeApo1SigUp_au, discInnerEdgeApo1SigDown_au, apltError_au)
	
	mPlt1SigUp_mJup = mPlt_mJup * fracErr1SigUp
	mPlt1SigDown_mJup = mPlt_mJup * fracErr1SigDown
	
	return mPlt_mJup, mPlt1SigUp_mJup, mPlt1SigDown_mJup
	
#------------------------------------------------------------------------
def GetLaplaceCoefficient(j, s, alpha):
	'''Returns b_s^j(alpha)'''
	
	laplaceCoefficientIntegrand = lambda psi: np.cos(j*psi) / (1.0 - 2*alpha*np.cos(psi) + alpha**2)**s
	
	laplaceCoefficient = 1.0 / np.pi * integrate.quad(laplaceCoefficientIntegrand, 0, 2*np.pi)[0]
	
	return laplaceCoefficient

#------------------------------------------------------------------------
def GetPearce2014MinimumPlanetMassSemimajorAxisEccentricityErrors_mJupAndAu(minMassPlanetPars, debugMode=False):
	'''~Gaussian error propagation on the Pearce & Wyatt (2022) minimum 
	plt mass, its semimajor axis and eccentricity. Only approximate,
	since really only valid if errors Gaussian and symmetric, but 
	probably ok for small fractional errors that are roughly symmetric.
	'''

	# Unpack the parameters of the minimum-mass plt
	mPlt_mJup = minMassPlanetPars['mPlt_mJup']
	pltSemimajorAxis_au = minMassPlanetPars['pltSemimajorAxis_au']
	pltEccentricity = minMassPlanetPars['pltEccentricity']

	# If the disc model is axisymmetric, set the plt eccentricity to 
	# zero for the following calculations (since it may have been set to 
	# be nan)
	if isDiscAsymmetric == False: pltEccentricity = 0.
	
	# Containers for different variables
	variableValues_unit = {}
	variableValues1SigUp_unit = {}
	variableValues1SigDown_unit = {}
	
	variableValues_unit['mStar_mSun'] = mStar_mSun
	variableValues1SigUp_unit['mStar_mSun'] = mStar1SigUp_mSun
	variableValues1SigDown_unit['mStar_mSun'] = mStar1SigDown_mSun
	
	variableValues_unit['age_myr'] = age_myr
	variableValues1SigUp_unit['age_myr'] = age1SigUp_myr
	variableValues1SigDown_unit['age_myr'] = age1SigDown_myr	
	
	variableValues_unit['discInnerEdgePeri_au'] = discInnerEdgePeri_au
	variableValues1SigUp_unit['discInnerEdgePeri_au'] = discInnerEdgePeri1SigUp_au
	variableValues1SigDown_unit['discInnerEdgePeri_au'] = discInnerEdgePeri1SigDown_au

	variableValues_unit['discInnerEdgeApo_au'] = discInnerEdgeApo_au
	variableValues1SigUp_unit['discInnerEdgeApo_au'] = discInnerEdgeApo1SigUp_au
	variableValues1SigDown_unit['discInnerEdgeApo_au'] = discInnerEdgeApo1SigDown_au
			
	differentialMPltWRTVariables_mJupPerUnit = {}
	differentialSemimajorAxisWRTVariables_auPerUnit = {}
	differentialEccentricityWRTVariables_PerUnit = {}

	# Derivatives of plt mass wrt star mass and age. These equations 
	# are valid for both asymmetric and axisymmetric models
	A = pltEccentricity / (3.-pltEccentricity) - 3.*discInnerEdgeApo_au / (discInnerEdgeApo_au*(1.+pltEccentricity) - pltSemimajorAxis_au*(1.+pltEccentricity)**2)

	differentialMPltWRTVariables_mJupPerUnit['mStar_mSun'] = mPlt_mJup/mStar_mSun * (1.-0.75*A) / (1.0 - A)
	differentialMPltWRTVariables_mJupPerUnit['age_myr'] = 0.5 * mPlt_mJup/age_myr * A / (1.0 - A)
	
	# Derivatives of plt mass wrt disc inner edge apocentre and 
	# pericentre. These differ for asymmetric and axisymmetric models. 
	# Start with axisymmetric model
	pltApocentre_au = pltSemimajorAxis_au*(1.+pltEccentricity)
	
	if isDiscAsymmetric:
		differentialMPltWRTVariables_mJupPerUnit['discInnerEdgePeri_au'] = 2./5.*mPlt_mJup/pltSemimajorAxis_au * (3.*discInnerEdgeApo_au/(discInnerEdgeApo_au*(1.+pltEccentricity)-pltSemimajorAxis_au*(1.+pltEccentricity)**2) + 1./(3.-pltEccentricity)) / (1.-A)
		differentialMPltWRTVariables_mJupPerUnit['discInnerEdgeApo_au'] = (0.25*mPlt_mJup/discInnerEdgeApo_au*A + 3.*mPlt_mJup/(discInnerEdgeApo_au-pltApocentre_au)*(1.-2./5.*discInnerEdgeApo_au/pltApocentre_au) - 2./5.*mPlt_mJup/(pltSemimajorAxis_au*(3.-pltEccentricity))) / (1.-A)
		
	else:
		differentialMPltWRTVariables_mJupPerUnit['discInnerEdgePeri_au'] = 0.
		differentialMPltWRTVariables_mJupPerUnit['discInnerEdgeApo_au'] = 9./4. * mPlt_mJup / (4.*discInnerEdgeApo_au - pltSemimajorAxis_au)

	# Differentials of plt semimajor axis wrt variables
	differentialSemimajorAxisWRTVariables_auPerUnit['mStar_mSun'] = pltSemimajorAxis_au*(differentialMPltWRTVariables_mJupPerUnit['mStar_mSun']/mPlt_mJup - 0.75/mStar_mSun)
	differentialSemimajorAxisWRTVariables_auPerUnit['age_myr'] = pltSemimajorAxis_au*(differentialMPltWRTVariables_mJupPerUnit['age_myr']/mPlt_mJup + 0.5/age_myr)
	differentialSemimajorAxisWRTVariables_auPerUnit['discInnerEdgePeri_au'] = pltSemimajorAxis_au*differentialMPltWRTVariables_mJupPerUnit['discInnerEdgePeri_au']/mPlt_mJup
	differentialSemimajorAxisWRTVariables_auPerUnit['discInnerEdgeApo_au'] = pltSemimajorAxis_au*(differentialMPltWRTVariables_mJupPerUnit['discInnerEdgeApo_au']/mPlt_mJup + 0.25/discInnerEdgeApo_au)

	# Differentials of plt eccentricity wrt variables
	if isDiscAsymmetric:
		differentialEccentricityWRTVariables_PerUnit['mStar_mSun'] = -pltEccentricity/pltSemimajorAxis_au*differentialSemimajorAxisWRTVariables_auPerUnit['mStar_mSun']
		differentialEccentricityWRTVariables_PerUnit['age_myr'] = -pltEccentricity/pltSemimajorAxis_au*differentialSemimajorAxisWRTVariables_auPerUnit['age_myr']
		differentialEccentricityWRTVariables_PerUnit['discInnerEdgePeri_au'] = -pltEccentricity/pltSemimajorAxis_au*differentialSemimajorAxisWRTVariables_auPerUnit['discInnerEdgePeri_au'] - 2./5./pltSemimajorAxis_au
		differentialEccentricityWRTVariables_PerUnit['discInnerEdgeApo_au'] = -pltEccentricity/pltSemimajorAxis_au*differentialSemimajorAxisWRTVariables_auPerUnit['discInnerEdgeApo_au'] + 2./5./pltSemimajorAxis_au

	# Get the total errors from the different sources
	mPlt1SigUp_mJup, mPlt1SigDown_mJup = GetGaussianAsymmetricUncertainties(variableValues1SigUp_unit, variableValues1SigDown_unit, differentialMPltWRTVariables_mJupPerUnit)
	pltSemimajorAxis1SigUp_au, pltSemimajorAxis1SigDown_au = GetGaussianAsymmetricUncertainties(variableValues1SigUp_unit, variableValues1SigDown_unit, differentialSemimajorAxisWRTVariables_auPerUnit)

	if isDiscAsymmetric:
		pltEccentricity1SigUp, pltEccentricity1SigDown = GetGaussianAsymmetricUncertainties(variableValues1SigUp_unit, variableValues1SigDown_unit, differentialEccentricityWRTVariables_PerUnit)
	
	else:
		pltEccentricity1SigUp, pltEccentricity1SigDown = np.nan, np.nan

	# Print if desired
	if debugMode:
		print(differentialMPltWRTVariables_mJupPerUnit)
		print(differentialSemimajorAxisWRTVariables_auPerUnit)
		print(differentialEccentricityWRTVariables_PerUnit)
		
		print(mPlt1SigUp_mJup, mPlt1SigDown_mJup, pltSemimajorAxis1SigUp_au, pltSemimajorAxis1SigDown_au, pltEccentricity1SigUp, pltEccentricity1SigDown)
	
	# Package the uncertainties up in a container
	uncertaintiesOnMinMassPlanetPars = {'mPlt_mJup': {'1SigUp': mPlt1SigUp_mJup, '1SigDown': mPlt1SigDown_mJup},
										'pltSemimajorAxis_au': {'1SigUp': pltSemimajorAxis1SigUp_au, '1SigDown': pltSemimajorAxis1SigDown_au},
										'pltEccentricity': {'1SigUp': pltEccentricity1SigUp, '1SigDown': pltEccentricity1SigDown}}
	
	return uncertaintiesOnMinMassPlanetPars

#------------------------------------------------------------------------
def GetErrorsOnPltMassFromHillConstraintAtSpecificSemimajorAxis_mJup(mPlt_mJup, pltSemimajorAxis_au, pltEccentricity, mStar_mSun, mStar1SigUp_mSun, mStar1SigDown_mSun, discInnerEdgePeri_au, discInnerEdgePeri1SigUp_au, discInnerEdgePeri1SigDown_au, discInnerEdgeApo_au, discInnerEdgeApo1SigUp_au, discInnerEdgeApo1SigDown_au, debugMode=False):
	'''Gaussian error propagation on plt mass limit from the apocentre
	of the inner disc edge being the plt apocentre plus 5 eccentric 
	Hill radii. Only approximate, since really only valid if errors 
	Gaussian and symmetric, but probably ok for small fractional errors
	that are roughly symmetric. Note this is the EXACT (not fractional) 
	error on the plt mass.'''
	
	# Containers for different variables
	variableValues1SigUp_unit = {}
	variableValues1SigDown_unit = {}
	
	variableValues1SigUp_unit['mStar_mSun'] = mStar1SigUp_mSun
	variableValues1SigDown_unit['mStar_mSun'] = mStar1SigDown_mSun
	
	variableValues1SigUp_unit['discInnerEdgePeri_au'] = discInnerEdgePeri1SigUp_au
	variableValues1SigDown_unit['discInnerEdgePeri_au'] = discInnerEdgePeri1SigDown_au

	variableValues1SigUp_unit['discInnerEdgeApo_au'] = discInnerEdgeApo1SigUp_au
	variableValues1SigDown_unit['discInnerEdgeApo_au'] = discInnerEdgeApo1SigDown_au
	
	differentialMPltWRTVariables_mJupPerUnit = {}

	# Derivatives of plt mass wrt star mass and age. These equations 
	# are valid for both asymmetric and axisymmetric models
	differentialMPltWRTVariables_mJupPerUnit['mStar_mSun'] = mPlt_mJup/mStar_mSun

	# Derivatives of plt mass wrt disc inner edge apocentre and 
	# pericentre. These differ for asymmetric and axisymmetric models. 
	if isDiscAsymmetric:
		differentialMPltWRTVariables_mJupPerUnit['discInnerEdgePeri_au'] = 2./5.*mPlt_mJup/pltSemimajorAxis_au * (3.*discInnerEdgeApo_au/(discInnerEdgeApo_au*(1.+pltEccentricity) - pltSemimajorAxis_au*(1.+pltEccentricity)**2) + 1./(3.-pltEccentricity))
		differentialMPltWRTVariables_mJupPerUnit['discInnerEdgeApo_au'] = 3.*mPlt_mJup/(discInnerEdgeApo_au-pltSemimajorAxis_au*(1.+pltEccentricity))*(1.-2./5.*discInnerEdgeApo_au/(pltSemimajorAxis_au*(1.+pltEccentricity))) - 2./5.*mPlt_mJup/(pltSemimajorAxis_au*(3.-pltEccentricity))
		
	else:
		differentialMPltWRTVariables_mJupPerUnit['discInnerEdgePeri_au'] = 0.
		differentialMPltWRTVariables_mJupPerUnit['discInnerEdgeApo_au'] = 3.*mPlt_mJup / (discInnerEdgeApo_au - pltSemimajorAxis_au)

	# Get the total errors from the different sources
	mPlt1SigUp_mJup, mPlt1SigDown_mJup = GetGaussianAsymmetricUncertainties(variableValues1SigUp_unit, variableValues1SigDown_unit, differentialMPltWRTVariables_mJupPerUnit)

	# Debug options
	if debugMode:
		print(mPlt_mJup, mPlt1SigUp_mJup, mPlt1SigDown_mJup, pltSemimajorAxis_au)
		print(differentialMPltWRTVariables_mJupPerUnit)
		
	return mPlt1SigUp_mJup, mPlt1SigDown_mJup
			
#------------------------------------------------------------------------
def GetFracErrorOnPltMassAtSemimajorAxisFromDiffusionTimeConstraint(mStar_mSun, mStar1SigUp_mSun, mStar1SigDown_mSun, age_myr, age1SigUp_myr, age1SigDown_myr, outerDiscInnerEdge_au, outerDiscInnerEdge1SigUp_au, outerDiscInnerEdge1SigDown_au, apltError_au=0):
	'''Gaussian error propagation on plt mass limit from the star age 
	being at least 10 times the diffusion timescale, for a plt at a 
	given semimajor axis. Only approximate, since really only valid if 
	errors Gaussian and symmetric, but probably ok for small fractional 
	errors that are roughly symmetric. Note this is the FRACTIONAL error 
	on the plt mass.'''

	if apltError_au != 0:
		print('+++ WARNING: Diffusion constraint not yet set up for errors on aplt')

	# Error contributions
	ageErr1SigUpCont = (-0.5 * age1SigDown_myr / age_myr)**2
	ageErr1SigDownCont = (-0.5 * age1SigUp_myr / age_myr)**2
	
	mStar1SigUpCont = (0.75 * mStar1SigUp_mSun / mStar_mSun)**2
	mStar1SigDownCont = (0.75 * mStar1SigDown_mSun / mStar_mSun)**2
	
	outerDisc1SigUpCont = (-0.25 * outerDiscInnerEdge1SigDown_au / outerDiscInnerEdge_au)**2
	outerDisc1SigDownCont = (-0.25 * outerDiscInnerEdge1SigUp_au / outerDiscInnerEdge_au)**2

	# Get the error on the plt mass from the diffusion timescale 
	# argument. Note that plt mass limit is larger for smaller outer
	# disc inner edges, so the 1 sig up error on plt mass uses the 1
	# sig down error on outer disc inner edge
	mPltFromDiffusion1SigUpFrac = (ageErr1SigUpCont + mStar1SigUpCont + outerDisc1SigUpCont)**.5
	mPltFromDiffusion1SigDownFrac = (ageErr1SigDownCont + mStar1SigDownCont + outerDisc1SigDownCont)**.5
		
	return mPltFromDiffusion1SigUpFrac, mPltFromDiffusion1SigDownFrac

#------------------------------------------------------------------------
def GetFracErrorOnPltMassAtSemimajorAxisFromSecularTimeConstraint(alpha, b1_32, mStar_mSun, mStar1SigUp_mSun, mStar1SigDown_mSun, age_myr, age1SigUp_myr, age1SigDown_myr, discInnerEdgeApo_au, discInnerEdgeApo1SigUp_au, discInnerEdgeApo1SigDown_au, apltError_au=0):
	'''Gaussian error propagation on plt mass limit from the star age 
	being at least 10 times the secular timescale, at a plt at a given 
	semimajor axis. Only approximate, since really only valid if errors
	Gaussian and symmetric, but probably ok for small fractional errors 
	that are roughly symmetric. Note this is the FRACTIONAL error on the
	plt mass'''
	
	if apltError_au != 0:
		print('+++ WARNING: Secular constraint not yet set up for errors on aplt')
		
	# Asymmetric error contribution from outer disc edge. Need to first 
	# decide whether mp increases or decreases with disc radius. If 
	# increasing with radius, then the 1 sig up mass error uses the 1 sig
	# up radius error. Otherwise, it uses the 1 sig down radius error
	laplaceDerivativeWRTAlpha = GetDerivativeOfLaplaceCoefficientWRTAlpha(1.0, 1.5, alpha)

	mPltDerivativeWRTRadius_mPlt = discInnerEdgeApo_au**-1 * (2.5 + alpha/b1_32 * laplaceDerivativeWRTAlpha)
	
	if mPltDerivativeWRTRadius_mPlt >= 0:
		outerDisc1SigUpCont = (mPltDerivativeWRTRadius_mPlt * discInnerEdgeApo1SigUp_au)**2
		outerDisc1SigDownCont = (mPltDerivativeWRTRadius_mPlt * discInnerEdgeApo1SigDown_au)**2

	else:
		outerDisc1SigUpCont = (mPltDerivativeWRTRadius_mPlt * discInnerEdgeApo1SigDown_au)**2		
		outerDisc1SigDownCont = (mPltDerivativeWRTRadius_mPlt * discInnerEdgeApo1SigUp_au)**2		

	age1SigUpCont = (-age1SigDown_myr / age_myr)**2
	age1SigDownCont = (-age1SigUp_myr / age_myr)**2	
	
	mStar1SigUpCont = (0.5 * mStar1SigUp_mSun / mStar_mSun)**2
	mStar1SigDownCont = (0.5 * mStar1SigDown_mSun / mStar_mSun)**2

	# Get the fractional error on the plt mass from the secular 
	# timescale argument
	mPltFromSecular1SigUpFrac = (age1SigUpCont + mStar1SigUpCont + outerDisc1SigUpCont)**.5
	mPltFromSecular1SigDownFrac = (age1SigDownCont + mStar1SigDownCont + outerDisc1SigDownCont)**.5
	
	return mPltFromSecular1SigUpFrac, mPltFromSecular1SigDownFrac
	
#------------------------------------------------------------------------
def GetDerivativeOfLaplaceCoefficientWRTAlpha(j, s, alpha):
	'''Returns d/dalpha b_s^j(alpha). Used for uncertainty calculations 
	on the Laplace coefficient. Note as alpha gets close to 1, the
	integral gets slowly convergent as it tends to infinity as alpha->1.'''
	
	laplaceCoefficientDerivativeIntegrand = lambda psi: -2*s*np.cos(j*psi) * (alpha-np.cos(psi)) / (1.0 - 2*alpha*np.cos(psi) + alpha**2)**(s+1)
	
	laplaceCoefficientDerivative = 1.0 / np.pi * integrate.quad(laplaceCoefficientDerivativeIntegrand, 0, 2*np.pi)[0]
			
	return laplaceCoefficientDerivative
				
############################ Print functions ############################
def CheckUserInputsOK():
	'''Check the user inputs are OK. All values should be positive, and 
	all parameters non-zero (although uncertainties can be zero)'''
	
	areUserInputsOK = True
	reasonsUnputsAreBad = []
	
	# All values must be positive and non-zero
	for value in [mStar_mSun, age_myr, discInnerEdgePeri_au, discInnerEdgeApo_au]:
		if math.isnan(value) or value <= 0:
			areUserInputsOK = False
			reasonsUnputsAreBad.append('All parameters should be non-zero, positive, and not nan (although uncertainties can be zero or nan)')
			break

	# All uncertainties must be positive or zero
	for uncertaintyValue in [mStar1SigUp_mSun, mStar1SigDown_mSun, age1SigUp_myr, age1SigDown_myr, discInnerEdgePeri1SigUp_au, discInnerEdgePeri1SigDown_au,
		discInnerEdgeApo1SigUp_au, discInnerEdgeApo1SigDown_au]:
		if math.isnan(uncertaintyValue) == False and uncertaintyValue < 0:
			reasonsUnputsAreBad.append('All uncertainties must each be either zero, positive or nan')
			areUserInputsOK = False			
			break
	
	# Disc apo and peri must be defined correctly
	if discInnerEdgeApo_au < discInnerEdgePeri_au:
		reasonsUnputsAreBad.append('Disc apocentre and pericentre are defined the wrong way around')
		areUserInputsOK = False		
			
	# Warn the user if the inputs are bad
	if areUserInputsOK == False:
		print('***ERROR*** Problem(s) with user inputs:')
		for reasonUnputsAreBad in reasonsUnputsAreBad:
			print('     -%s' % reasonUnputsAreBad)
		print()
		
	return areUserInputsOK
	
#------------------------------------------------------------------------
def PrintUserInputs():
	'''Print the user inputs'''
	
	print('User inputs:')
	print('     Star mass: %s MSun' % GetValueAndUncertaintyString(mStar_mSun, mStar1SigUp_mSun, mStar1SigDown_mSun))
	print('     System age: %s myr' % GetValueAndUncertaintyString(age_myr, age1SigUp_myr, age1SigDown_myr))
	print('     Disc inner edge peri: %s au' % GetValueAndUncertaintyString(discInnerEdgePeri_au, discInnerEdgePeri1SigUp_au, discInnerEdgePeri1SigDown_au))
	print('     Disc inner edge apo: %s au' % GetValueAndUncertaintyString(discInnerEdgeApo_au, discInnerEdgeApo1SigUp_au, discInnerEdgeApo1SigDown_au))
	print()

#------------------------------------------------------------------------
def PrintProgramOutputs(minMassPlanetPars, uncertaintiesOnMinMassPlanetPars):
	'''Print the user inputs'''
	
	# Unpack the parameters of the minimum-mass plt and its 
	# uncertainties
	mPlt_mJup = minMassPlanetPars['mPlt_mJup']
	pltSemimajorAxis_au = minMassPlanetPars['pltSemimajorAxis_au']
	pltEccentricity = minMassPlanetPars['pltEccentricity']
	mPlt1SigUp_mJup = uncertaintiesOnMinMassPlanetPars['mPlt_mJup']['1SigUp']
	mPlt1SigDown_mJup = uncertaintiesOnMinMassPlanetPars['mPlt_mJup']['1SigDown']
	pltSemimajorAxis1SigUp_au = uncertaintiesOnMinMassPlanetPars['pltSemimajorAxis_au']['1SigUp']
	pltSemimajorAxis1SigDown_au = uncertaintiesOnMinMassPlanetPars['pltSemimajorAxis_au']['1SigDown']
	pltEccentricity1SigUp = uncertaintiesOnMinMassPlanetPars['pltEccentricity']['1SigUp']
	pltEccentricity1SigDown = uncertaintiesOnMinMassPlanetPars['pltEccentricity']['1SigDown']	
	
	print('Results:')
	print('     Min. mass of plt to truncate disc: %s MJup' % GetValueAndUncertaintyString(mPlt_mJup, mPlt1SigUp_mJup, mPlt1SigDown_mJup))
	print('     Max. semimajor axis of plt to truncate disc: %s au' % GetValueAndUncertaintyString(pltSemimajorAxis_au, pltSemimajorAxis1SigUp_au, pltSemimajorAxis1SigDown_au))
	
	# If disc axisymmetric, no eccentricitiy lower limit
	if discInnerEdgeApo_au == discInnerEdgePeri_au:
		print('     Disc axisymmetric, so no lower limit on plt eccentricity')
	
	# Otherwise, the disc is asymmetric and there is a lower limit
	else:
		print('     Min. eccentricity of plt to truncate disc: %s' % GetValueAndUncertaintyString(pltEccentricity, pltEccentricity1SigUp, pltEccentricity1SigDown))
	
	print()
	
#------------------------------------------------------------------------
def GetValueAndUncertaintyString(value, err1SigUp, err1SigDown):
	'''Get a string neatly showing the value and its uncertainties'''
	
	# The orders of the largest and smallest values that should be 
	# written in non-SI notation
	minOrderForNonSINotation = -3
	maxOrderForNonSINotation = 3
		
	# If the value is non-zero, and neither it nor its uncertainties are 
	# nans, proceed. Some of the functions below would otherwise fail
	if math.isnan(value) == False and math.isnan(err1SigUp) == False and math.isnan(err1SigDown) == False and value > 0:

		# Errors quoted to 1 sig fig
		err1SigUpRounded = abs(RoundNumberToDesiredSigFigs(err1SigUp, 1))
		err1SigDownRounded = abs(RoundNumberToDesiredSigFigs(err1SigDown, 1))

		# Get the order of the value and the smallest uncertainty
		orderOfValue = GetBase10OrderOfNumber(value)
		orderOfSmallestError = GetBase10OrderOfNumber(min(err1SigUpRounded, err1SigDownRounded))

		# Round the value to the correct number of significant figures,
		# such that the final figure is at the order of the error
		sigFigsToRoundValueTo = max(orderOfValue - orderOfSmallestError + 1, 1)
		valueRounded = RoundNumberToDesiredSigFigs(value, sigFigsToRoundValueTo)
		orderOfRoundedValue = GetBase10OrderOfNumber(valueRounded)

		# If the rounded value has gone up an order, will round value 
		# to an extra significant figure (e.g. 0.099 +/- 0.03 -> 0.10 +/- 0.03)
		if orderOfRoundedValue > orderOfValue:
			sigFigsToRoundValueTo += 1
		
		# If the value is very small or large, divide it by its order, 
		# and later quote order in string. Use rounding function to 
		# remove rounding errors, and note that uncertainties are 
		# always quoted to 1 sig fig
		if orderOfRoundedValue > maxOrderForNonSINotation or orderOfRoundedValue < minOrderForNonSINotation:
			wasPowerAdjustmentDone = True
			
			#valueRounded /= 10**orderOfRoundedValue
			valueRounded = RoundNumberToDesiredSigFigs(valueRounded/10**orderOfRoundedValue, sigFigsToRoundValueTo)
			err1SigUpRounded = RoundNumberToDesiredSigFigs(err1SigUpRounded/10**orderOfRoundedValue, 1)
			err1SigDownRounded  = RoundNumberToDesiredSigFigs(err1SigDownRounded/10**orderOfRoundedValue, 1)
			
			orderOfRoundedValueAfterPowerAdjust = MF.GetBase10OrderOfNumber(valueRounded)
			orderOfSmallestRoundedErrorAfterPowerAdjust = MF.GetBase10OrderOfNumber(min(err1SigUpRounded, err1SigDownRounded))
			
		else:
			wasPowerAdjustmentDone = False
			orderOfRoundedValueAfterPowerAdjust = orderOfRoundedValue
			orderOfSmallestRoundedErrorAfterPowerAdjust = orderOfSmallestError

		# If all significant figures of both the value and 
		# uncertainties are to the left of the decimal point, the value
		# and uncertainties are integers
		numberOfValueFiguresLeftOfDecimalPoint = max(orderOfRoundedValueAfterPowerAdjust + 1, 0)
		numberOfErrorFiguresLeftOfDecimalPoint = orderOfSmallestRoundedErrorAfterPowerAdjust + 1

		if numberOfValueFiguresLeftOfDecimalPoint - sigFigsToRoundValueTo >= 0:
			valueRounded = int(valueRounded)
			err1SigUpRounded = int(err1SigUpRounded)
			err1SigDownRounded = int(err1SigDownRounded)

		# Convert the value to a string. If there are value figures to 
		# the right of the decimal point, and the final one(s) should 
		# be zero, append zeros to the string 
		valueRoundedString = str(valueRounded)
		
		numberOfValueFiguresNeededRightOfDecimalPoint = max(sigFigsToRoundValueTo - (orderOfRoundedValueAfterPowerAdjust+1), 0)
		
		if numberOfValueFiguresNeededRightOfDecimalPoint > 0:

			while True:
				indexOfPointInString = valueRoundedString.index('.')
				numberOfValueFiguresRightOfDecimalPointInString = len(valueRoundedString[indexOfPointInString+1:])
									
				if numberOfValueFiguresRightOfDecimalPointInString == numberOfValueFiguresNeededRightOfDecimalPoint:
					break
				
				valueRoundedString += '0'				
		
		# If errors are symmetric
		if err1SigUpRounded == err1SigDownRounded:
			if wasPowerAdjustmentDone:
				valueAndUncertaintyString = '(%s +/- %s) * 10^%s' % (valueRoundedString, err1SigUpRounded, orderOfRoundedValue)

			else:
				valueAndUncertaintyString = '%s +/- %s' % (valueRoundedString, err1SigUpRounded)

		# Otherwise errors are asymmetric
		else:
			if wasPowerAdjustmentDone:
				valueAndUncertaintyString = '(%s +%s -%s) * 10^%s' % (valueRoundedString, err1SigUpRounded, err1SigDownRounded, orderOfRoundedValue)	
			
			else:
				valueAndUncertaintyString = '%s +%s -%s' % (valueRoundedString, err1SigUpRounded, err1SigDownRounded)
					
	# Otherwise value is zero or nan, or at least one error is Nan
	else:
		valueAndUncertaintyString = '%s +%s -%s' % (value, err1SigUp, err1SigDown)

	return valueAndUncertaintyString

############################# Plot functions ############################
def MakePlot(minMassPlanetPars, uncertaintiesOnMinMassPlanetPars):
	'''Plot the constraints on plt mass and semimajor axis, like Fig. 7
	in Pearce et al. 2022 or Fig. 17 in	Pearce & Wyatt 2014.'''

	print('Making plot...')

	# Unpack the parameters of the minimum-mass plt and its 
	# uncertainties
	mPltMinForTrunc_mJup = minMassPlanetPars['mPlt_mJup']
	pltSemimajorAxisMaxForTrunc_au = minMassPlanetPars['pltSemimajorAxis_au']
	pltEccentricityMinForTrunc = minMassPlanetPars['pltEccentricity']
	mPltMinForTrunc1SigUp_mJup = uncertaintiesOnMinMassPlanetPars['mPlt_mJup']['1SigUp']
	mPltMinForTrunc1SigDown_mJup = uncertaintiesOnMinMassPlanetPars['mPlt_mJup']['1SigDown']
	pltSemimajorAxisMaxForTrunc1SigUp_au = uncertaintiesOnMinMassPlanetPars['pltSemimajorAxis_au']['1SigUp']
	pltSemimajorAxisMaxForTrunc1SigDown_au = uncertaintiesOnMinMassPlanetPars['pltSemimajorAxis_au']['1SigDown']
	pltEccentricityMinForTrunc1SigUp = uncertaintiesOnMinMassPlanetPars['pltEccentricity']['1SigUp']
	pltEccentricityMinForTrunc1SigDown = uncertaintiesOnMinMassPlanetPars['pltEccentricity']['1SigDown']	
	
	# Get the maximum possible semimajor axis of the planet. This will be
	# where the planet apocentre exceeds the disc inner edge apocentre. 
	# From rearranging Equation 5 in Pearce et al. (2022)
	maxPossiblePltSemimajorAxis_au = (3.*discInnerEdgeApo_au + 2.*discInnerEdgePeri_au)/5.
	maxPossiblePltSemimajorAxis1SigUp_au = ((3./5.*discInnerEdgeApo1SigUp_au)**2 + (2./5.*discInnerEdgePeri1SigUp_au)**2)**0.5
	maxPossiblePltSemimajorAxis1SigDown_au = ((3./5.*discInnerEdgeApo1SigDown_au)**2 + (2./5.*discInnerEdgePeri1SigDown_au)**2)**0.5	

	# If disc asymmetric, get the minimum possible semimajor axis of the 
	# planet. This is where the planet would need an eccentricity >=1 to 
	# drive the disc eccentricity (according to our model), which is 
	# unphysical. Limit from rearranging Equation 5 in Pearce et al. 2022
	minPossiblePltSemimajorAxis_au = 2./5.*(discInnerEdgeApo_au - discInnerEdgePeri_au)
	minPossiblePltSemimajorAxis1SigUp_au = 2./5.*(discInnerEdgeApo1SigUp_au**2 + discInnerEdgePeri1SigDown_au**2)**0.5
	minPossiblePltSemimajorAxis1SigDown_au = 2./5.*(discInnerEdgeApo1SigDown_au**2 + discInnerEdgePeri1SigUp_au**2)**0.5
	minPossiblePltSemimajorAxis_au = max(minPossiblePltSemimajorAxis_au, 1e-99)

	# For each possible plt semimajor axis, run the 
	# Pearce et al. (2022) analysis (including uncertainties)
	numberOfSemimajorAxisSteps = 1000
	auStep = (maxPossiblePltSemimajorAxis_au - minPossiblePltSemimajorAxis_au)/ float(numberOfSemimajorAxisSteps)
	as_au = np.arange(minPossiblePltSemimajorAxis_au, maxPossiblePltSemimajorAxis_au*.999 + auStep, auStep)

	mPltsFromHill_MJup, mPltsFromDiffusionTime_MJup, mPltsFromSecularTime_MJup, mPltsFromStirring_MJup = [], [], [], []
	mPltsFromHill1SigUpVal_MJup, mPltsFromHill1SigDownVal_MJup, mPltsFromDiffusionTime1SigUpVal_MJup, mPltsFromDiffusionTime1SigDownVal_MJup, mPltsFromSecularTime1SigUpVal_MJup, mPltsFromSecularTime1SigDownVal_MJup, mPltsFromStirring1SigUpVal_MJup, mPltsFromStirring1SigDownVal_MJup = [], [], [], [], [], [], [], []

	for a_au in as_au:
	
		# Hill constraint
		mPltFromHill_MJup, pltEccentricityFromHill = GetPltMassAndEccentricityFromHillConstraint_mJup(a_au, mStar_mSun, discInnerEdgePeri_au, discInnerEdgeApo_au)
		mPltFromHill1SigUp_MJup, mPltFromHill1SigDown_MJup = GetErrorsOnPltMassFromHillConstraintAtSpecificSemimajorAxis_mJup(mPltFromHill_MJup, a_au, pltEccentricityFromHill, mStar_mSun, mStar1SigUp_mSun, mStar1SigDown_mSun, discInnerEdgePeri_au, discInnerEdgePeri1SigUp_au, discInnerEdgePeri1SigDown_au, discInnerEdgeApo_au, discInnerEdgeApo1SigUp_au, discInnerEdgeApo1SigDown_au)
		mPltFromHill1SigUpVal_MJup = mPltFromHill_MJup + mPltFromHill1SigUp_MJup
		mPltFromHill1SigDownVal_MJup = mPltFromHill_MJup - mPltFromHill1SigDown_MJup

		if math.isnan(mPltFromHill1SigUpVal_MJup): mPltFromHill1SigUpVal_MJup = 1e-99
		if math.isnan(mPltFromHill1SigDownVal_MJup): mPltFromHill1SigDownVal_MJup = 1e-99

		# To avoid errors on log plot, set zero value to very small
		# positive value
		if mPltFromHill_MJup == 0:
			mPltFromHill_MJup, mPltFromHill1SigUp_MJup, mPltFromHill1SigDown_MJup = 1e-99, 1e-99, 1e-99
		
		# Diffusion time
		mPltFromDiffusionTime_MJup = GetPltMassFromDiffusionTimeConstraint_mJup(a_au, mStar_mSun, age_myr, discInnerEdgeApo_au)
		mPltFromDiffusion1SigUpFrac, mPltFromDiffusion1SigDownFrac = GetFracErrorOnPltMassAtSemimajorAxisFromDiffusionTimeConstraint(mStar_mSun, mStar1SigUp_mSun, mStar1SigDown_mSun, age_myr, age1SigUp_myr, age1SigDown_myr, discInnerEdgeApo_au, discInnerEdgeApo1SigUp_au, discInnerEdgeApo1SigDown_au)
		mPltFromDiffusionTime1SigUpVal_MJup = mPltFromDiffusionTime_MJup * (1 + mPltFromDiffusion1SigUpFrac)
		mPltFromDiffusionTime1SigDownVal_MJup = mPltFromDiffusionTime_MJup * (1 - mPltFromDiffusion1SigDownFrac)
		
		# Secular time
		mPltFromSecularTimeConstraint_MJup, mPltFromSecularTime1SigUp_MJup, mPltFromSecularTime1SigDown_MJup = GetPltMassAndAbsErrorsFromSecularTimeConstraint_mJup(a_au, mStar_mSun, age_myr, discInnerEdgeApo_au, mStar1SigUp_mSun, mStar1SigDown_mSun, age1SigUp_myr, age1SigDown_myr, discInnerEdgeApo1SigUp_au, discInnerEdgeApo1SigDown_au)
		mPltFromSecularTimeConstraint1SigUpVal_MJup = mPltFromSecularTimeConstraint_MJup + mPltFromSecularTime1SigUp_MJup
		mPltFromSecularTimeConstraint1SigDownVal_MJup = mPltFromSecularTimeConstraint_MJup - mPltFromSecularTime1SigDown_MJup

		'''ADD MASS NEEDED TO STIR HERE, WHICH DEPENDS ON ECCENTRICITY? 
		ASSUME SAME PLANET BOTH SCULPTS AND STIRS DISC'''

		# Add values to lists
		mPltsFromHill_MJup.append(mPltFromHill_MJup)
		mPltsFromHill1SigUpVal_MJup.append(mPltFromHill1SigUpVal_MJup)
		mPltsFromHill1SigDownVal_MJup.append(mPltFromHill1SigDownVal_MJup)		
		
		mPltsFromDiffusionTime_MJup.append(mPltFromDiffusionTime_MJup)
		mPltsFromDiffusionTime1SigUpVal_MJup.append(mPltFromDiffusionTime1SigUpVal_MJup)
		mPltsFromDiffusionTime1SigDownVal_MJup.append(mPltFromDiffusionTime1SigDownVal_MJup)
		
		mPltsFromSecularTime_MJup.append(mPltFromSecularTimeConstraint_MJup)
		mPltsFromSecularTime1SigUpVal_MJup.append(mPltFromSecularTimeConstraint1SigUpVal_MJup)
		mPltsFromSecularTime1SigDownVal_MJup.append(mPltFromSecularTimeConstraint1SigDownVal_MJup)

	# Now produce the plot

	# Vertical line for disc inner edge (apocentre)
	if isDiscAsymmetric:
		discApoLabel = 'Disc inner edge apo.'
		discApoLw = 0.5
	else:
		discApoLabel = 'Disc inner edge'
		discApoLw = 1
	
	plt.axvline(discInnerEdgeApo_au, color='k', lw=discApoLw, ls='-.', label=discApoLabel, zorder=100)
	plt.axvspan(discInnerEdgeApo_au, 1e99, color='#bf7171', zorder=0)
	plt.axvspan(discInnerEdgeApo_au - discInnerEdgeApo1SigDown_au, discInnerEdgeApo_au + discInnerEdgeApo1SigUp_au, color = 'grey', alpha=0.4, zorder=50)

	# Planet masses from Hill constraint
	plt.plot(as_au, mPltsFromHill_MJup, 'k', label='Hill', zorder=100)
	plt.fill_between(as_au, mPltsFromHill1SigDownVal_MJup, mPltsFromHill1SigUpVal_MJup, lw=0, color = 'grey', alpha=0.4, zorder=30)
	plt.fill_between(as_au, mPltsFromHill1SigUpVal_MJup, 1e99, lw=0, color='#b3b3b3', zorder=50)
	plt.fill_between(as_au, mPltsFromHill1SigDownVal_MJup, lw=0, color='#b3b3b3', zorder=50)

	# Planet masses from diffusion time constraint
	plt.plot(as_au, mPltsFromDiffusionTime_MJup, 'r:', label='Diffusion', zorder=100)
	plt.fill_between(as_au, mPltsFromDiffusionTime1SigDownVal_MJup, mPltsFromDiffusionTime1SigUpVal_MJup, lw=0, color = 'grey', alpha=0.4, zorder=50)
	plt.fill_between(as_au, mPltsFromDiffusionTime1SigUpVal_MJup, 1e99, lw=0, color='w', zorder=40)
	plt.fill_between(as_au, mPltsFromDiffusionTime1SigDownVal_MJup, lw=0, color='#b3b3b3', zorder=20)
	
	# Planet masses from secular time constraint
	plt.plot(as_au, mPltsFromSecularTime_MJup, 'b:', label='Secular', zorder=100)	
	plt.fill_between(as_au, mPltsFromSecularTime1SigDownVal_MJup, mPltsFromSecularTime1SigUpVal_MJup, lw=0, color = 'grey', alpha=0.4, zorder=50)
	plt.fill_between(as_au, mPltsFromSecularTime1SigDownVal_MJup, lw=0, color='#b3b3b3', zorder=20)	

	# Minimum-mass planet
	pltSemimajorAxisMaxForTruncErrors_au = np.zeros((2, 1))
	pltSemimajorAxisMaxForTruncErrors_au[0] = pltSemimajorAxisMaxForTrunc1SigDown_au
	pltSemimajorAxisMaxForTruncErrors_au[1] = pltSemimajorAxisMaxForTrunc1SigUp_au	
	
	mPltMinForTruncErrors_MJup = np.zeros((2, 1))	
	mPltMinForTruncErrors_MJup[0] = mPltMinForTrunc1SigDown_mJup
	mPltMinForTruncErrors_MJup[1] = mPltMinForTrunc1SigUp_mJup
	plt.errorbar(pltSemimajorAxisMaxForTrunc_au, mPltMinForTrunc_mJup, xerr = pltSemimajorAxisMaxForTruncErrors_au, yerr = mPltMinForTruncErrors_MJup, linestyle = '', marker = 'o', color='r', ecolor='k', elinewidth = 0.5, capsize = 3, zorder = 200, label='Min. mass planet')	

	# Vertical line for max and min planet semimajor axis (before planet 
	# apocentre would lie outside disc inner edge apocentre). Only plot 
	# if disc asymmetric, since otherwise the planet eccentricity is zero
	# and this bound is just the disc inner edge
	if isDiscAsymmetric:
		plt.axvline(maxPossiblePltSemimajorAxis_au, color='k', ls= '--', label=r'$Q_{plt} < Q_{i}$', zorder=100)
		plt.axvspan(maxPossiblePltSemimajorAxis_au, 1e99, color='#bf7171', zorder=0)
		plt.axvspan(maxPossiblePltSemimajorAxis_au - maxPossiblePltSemimajorAxis1SigDown_au, maxPossiblePltSemimajorAxis_au + maxPossiblePltSemimajorAxis1SigUp_au, color = 'grey', alpha=0.4, zorder=50)

		plt.axvline(minPossiblePltSemimajorAxis_au, color='k', ls= (0, (3, 4, 1, 4, 1, 4)), label=r'$e_{plt} < 1$', zorder=100)
		plt.axvspan(0, minPossiblePltSemimajorAxis_au, color='#b3b3b3', zorder=0)
		plt.axvspan(minPossiblePltSemimajorAxis_au - minPossiblePltSemimajorAxis1SigDown_au, minPossiblePltSemimajorAxis_au + minPossiblePltSemimajorAxis1SigUp_au, color = 'grey', alpha=0.4, zorder=50)

	# Legend
	plt.legend().set_zorder(1000)

	# Graph settings
	plt.xlabel(r'Planet semimajor axis / au')
	plt.ylabel(r'Planet mass / ${\rm M_{Jup}}$')
	
	# Axis limits	
	yAxisMin = 0.01*(mPltMinForTrunc_mJup - mPltMinForTrunc1SigDown_mJup)
	if yAxisMin<=0: yAxisMin = 0.01*mPltMinForTrunc_mJup

	yAxisMax = 100*(mPltMinForTrunc_mJup + mPltMinForTrunc1SigUp_mJup)

	plt.xlim(0, 1.1*(discInnerEdgeApo_au+discInnerEdgeApo1SigUp_au))
	plt.ylim(yAxisMin, yAxisMax)	

	# Log scale (must be after axis limits)
	plt.yscale('log')
			
	# Show the plot	
	plt.show()
	
	print('Complete')
	print()
	
################################ Program ################################
print()

# Print the user inputs
PrintUserInputs()
	
# Check user inputs fine
areUserInputsOK = CheckUserInputsOK()

# Continue if the user inputs are OK
if areUserInputsOK:

	# Get the parameters of the minimum-mass plt that can truncate the
	# disc
	minMassPlanetPars = GetPearce2022MinimumPlanetMassAndLocationAndEccentricity_mJupAndAu()

	# Get the associated uncertainties
	uncertaintiesOnMinMassPlanetPars = GetPearce2014MinimumPlanetMassSemimajorAxisEccentricityErrors_mJupAndAu(minMassPlanetPars)

	# Print the results
	PrintProgramOutputs(minMassPlanetPars, uncertaintiesOnMinMassPlanetPars)

	# Make the figure if desired
	if shouldPlotBeMade:
		MakePlot(minMassPlanetPars, uncertaintiesOnMinMassPlanetPars)

#########################################################################

