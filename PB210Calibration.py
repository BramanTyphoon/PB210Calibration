#........................Lead-210 dating class and functions....................
# Author: James Bramante
# Date: September 5, 2019
#
# This class uses Monte Carlo simulations to produce estimates of sediment age
# and sedimentation rate given Pb-210 gamma count values and some other inputs,
# as described below (make sure you get units correct!):
# model: Can be either (default = "crs"):
#           - constant initial concentration ("cic") model, as in
#               Donnelly and Bertness (2001) PNAS 98: 14218-14223
#           - constant rate of supply ("crs") model, as in
#               Appleby and Oldfield (1978) Catena 5: 1-8
#               or Appleby's 2001 book chapter
# activity: Bq/kg - counts from Gamma measurements and calculations
# error: Bq/kg - error for each activity data point (assumed to be std. dev.,1 sigma)
# depth: m - depth in core of each activity data point
# density: kg/m^3 - dry bulk density (dry weight/wet volume) of the sediment at each
#           depth point.  If unknown, put 0 to assume uniform bulk density
# bkgrd: m - depth in core below which you think Pb-210 is supported (i.e. baseline)
# pb214: Bq/kg - optional 214Pb counts for each activity count, to use as estimate
#                of supported 210Pb.
# pb214_error: Bq/kg - error for each pb214 data point
#

# Import necessary modules
import numpy as np
import scipy.integrate as scint
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as cfit


# Define the initialization of the class
class PB210Calibration():
    """A utility for calibrating sedimentation rates from Pb210 measurements"""

    # DEFAULT VARIABLES
    DEFAULT_MODEL = "crs"
    DEFAULT_ERROR = 0.03 # 3% error
    DEFAULT_DENSITY = 2000 # kgm^-3, dry bulk density at each sample depth
    DEFAULT_BKGRD = -1 # m, depth below which Pb-210 is supported
    #DEFAULT_FRGRD = 0 # m, depth above which redox gradients or bioturbation
                      # disrupt Pb-210 profile. Samples above this depth will be
                      # ignored. NOT CURRENTLY IMPLEMENTED
    DEFAULT_PB214 = ()
    DEFAULT_PB214_ERROR = ()

    # Class variables
    max_iter = 1000 # Maximum number of pulls for each Monte Carlo simulation
    halflife = 22.26 # years, half-life of Pb-210
    decay = np.log(2)/halflife # yr^-1, decay constant for Pb-210
    initial_depo_rate = 1 # kgyr^-1, initial guess at sedimentation rate
    initial_depo_intercept = 0 # Bq/kg, initial guess at background rate if unknown
    integrate_method = 'exponential' #Integration to use. Either 'exponential'
                                     #or 'linear'

    def __init__(self, activity, error, depth, density = DEFAULT_DENSITY, \
                bkgrd = DEFAULT_BKGRD,model = DEFAULT_MODEL,pb214 = DEFAULT_PB214,\
                pb214_error = DEFAULT_PB214_ERROR):
        self.data = dict()
        self.model = model
        self.data['activity'] = np.array(activity)
        self.data['depth'] = np.array(depth)
        self.data['pb214'] = np.array(pb214)
        self.data['pb214Error'] = np.array(pb214_error)
        if hasattr(density, "__len__"):
            self.data['density'] = np.array(density)
        else:
            self.data['density'] = np.array([density])
        self.bkgrd = bkgrd
        self.data['error'] = np.array(error)
        if np.size(self.data['error'])<1:
            self.data['error'] = self.DEFAULT_ERROR * self.data['activity']
        self.organize() # Calculate derivative variables


    def organize(self):

        # Make sure data are in column arrays
        for dat in self.data:
            if self.data[dat].ndim > 1:
                self.data[dat] = self.data[dat].flatten()

        # Derive some additional variables
        self.num_data = np.size(self.data['depth'])
        if self.bkgrd < 0 and self.model=='crs':
            self.bkgrd = np.max(self.data['depth'])
        if np.size(self.data['density'])==1:
            self.data['density'] = self.data['density'][0] * \
                np.ones(np.shape(self.data['depth']))
        if np.size(self.data['pb214'])<1:
            self.data['pb214'] = np.zeros(np.shape(self.data['activity']))
            self.data['pb214Error'] = np.zeros(np.shape(self.data['activity']))
            
        # Define empty variable templates
        self.data['supported'] = np.ones((self.num_data,self.max_iter))*np.nan
        self.data['unsupported'] = np.ones((self.num_data,self.max_iter))*np.nan
        self.data['age'] = np.ones((self.num_data,self.max_iter))*np.nan
        self.data['rate'] = np.ones((self.num_data,self.max_iter))*np.nan
        self.data['inventory'] = np.ones((self.num_data,self.max_iter))*np.nan
        self.data['sortindex'] = np.argsort(self.data['depth'])

        # Sort array values by depth
        for dat in self.data:
            if not dat == 'sortindex':
                self.data[dat] = self.data[dat][self.data['sortindex']]


        # Calculate derivative variables
        self.decay = np.log(2) / self.halflife
        self.temp_unsupport = 0
        if not np.any(self.data['density']):
            self.data['mass'] = self.data['depth']
        else:
            if not self.data['depth'][0] == 0:
                self.data['mass'] = \
                    scint.cumtrapz(np.hstack((np.array(self.data['density'][0]),\
                    self.data['density'])),\
                    np.hstack((np.array([0]),self.data['depth'])))
            else:
                self.data['mass'] = np.hstack((np.zeros(1),scint.cumtrapz(\
                     self.data['density'],self.data['depth'])))




    def calibrate(self):
        # Populate empty data matrices with values drawn from Monte Carlo to
        # estimate supported Pb210 activity
        if (not self.bkgrd < 0):
            bkgrd_activity = self.data['activity'][self.data['depth']>=self.bkgrd]
            bkgrd_error = self.data['error'][self.data['depth']>=self.bkgrd]
            self.bkgrd_depth = self.data['depth']\
                    [np.min(np.where(self.data['depth']>=self.bkgrd))]
        else:
            bkgrd_activity = np.zeros(1)
            bkgrd_error = np.zeros(1)
            self.bkgrd_depth = np.max(self.data['depth'])

        if not np.any(self.data['pb214']):
            random_support = np.zeros((1,self.max_iter))
            for ii in np.arange(np.size(bkgrd_activity)):
                random_support += np.random.normal(bkgrd_activity[ii],bkgrd_error[ii],\
                (1,self.max_iter))/np.size(bkgrd_activity)
            self.data['supported'] = np.tile(random_support,(self.num_data,1))

        else:
            for ii in np.arange(self.num_data):
                self.data['supported'][ii,:] = \
                    np.random.normal(self.data['pb214'][ii],\
                    self.data['pb214Error'][ii],(1,self.max_iter))

        # Now perform random pulls of total activity data
        for ii in np.arange(self.num_data): # Randomly pull values of total Pb210
            self.data['unsupported'][ii,:] = \
                np.random.normal(self.data['activity'][ii],self.data['error'][ii],\
                (1,self.max_iter))
        self.data['unsupported'] -= self.data['supported'] # Subtract out supported
                                                           # Pb210

        # I'm not sure it's good form to set the hypothetical background values
        # to zero, as this changes the resulting age curve considerably near the
        # max utility age of Pb-210 dating (~100+ years). If using CRS, ages
        # greater than ~60 years should also be disregarded
        # if not self.bkgrd < 0:
        #   self.data['unsupported'][self.data['depth']>=self.bkgrd,:] = 0

        # Integrate the unsupported values to get inventory. Numerical method
        # used can be either linear (conventional cumtrapz) or exponential decay
        if self.integrate_method == 'exponential':
            temp_inventory = \
                self.log_cumtrapz(np.vstack((self.data['unsupported'][0,:], \
                self.data['unsupported'])),\
                np.tile(np.expand_dims(np.hstack((np.zeros(1),self.data['mass'])),1),\
                (1,self.max_iter)))
        else:
            temp_inventory = np.vstack((np.zeros((1,self.max_iter)),\
                scint.cumtrapz(np.vstack((self.data['unsupported'][0,:], \
                self.data['unsupported'])),\
                np.tile(np.expand_dims(np.hstack((np.zeros(1),self.data['mass'])),1),\
                (1,self.max_iter)),axis=0)))

        # Calculate A0 using equation in Appleby (2001)
        bkgrd_ind = np.nonzero(self.data['depth']==self.bkgrd_depth)[0][0]
        A0 = temp_inventory[bkgrd_ind-1,:] + 0.5 * \
                self.data['unsupported'][bkgrd_ind-1,:]* \
                (self.data['mass'][bkgrd_ind]-self.data['mass'][bkgrd_ind-1])
        self.data['inventory'] = A0 - temp_inventory


        # Now we perform one of two operations, given a particular model
        if self.model.lower() == 'crs':
            self.crs()
        elif self.model.lower() == 'cic':
            self.cic()


    def set_parameter(self,param,value):
        # Change model parameters
        if param.lower() in self.__dict__.keys:
            if param.lower()=='decay':
                self.halflife = np.log(2)/value
            self.__dict__[param.lower()] = value
            self.organize()
        else:
            raise AttributeError('The parameter name you have entered is invalid')

    def set_data(self,data,value):
        # Change model base data after initialization
        if data.lower() in self.data.keys:
            self.data[data.lower()] = value
            self.organize()
        else:
            raise AttributeError('The data name you have entered is invalid')

    def crs(self):
        # Perform constant rate of supply modeling of Pb-210 inventory profile
        # For full method, see Appleby and Oldfield (1978) Catena 5: 1-8 or
        # Sanchez-Cabena et al. (2000) Limnology and Oceanography 45: 990-995
        # or Appleby (2001) book chapter

        # Calculate age directly from the inventory data
        self.data['age'] = 1. / self.decay * np.log(\
        np.tile(self.data['inventory'][0,:],(self.num_data,1)) / \
        self.data['inventory'][1:,:])
        # Estimate initial/surface activity (turns out, this is actually just
        # equal to activity at the shallowest input depth)
        self.data['initial'] = 2*(self.data['inventory'][0,:] - \
            self.data['inventory'][1,:])/(np.ones((1,self.max_iter)) * \
            self.data['mass'][0]) - self.data['unsupported'][0,:]

        # Calculate time-varying deposition rate
        self.data['rate'] = self.decay * self.data['inventory'][1:,:] / \
            self.data['unsupported'] / \
            np.tile(np.expand_dims(self.data['density'],1),\
            (1,self.max_iter))

        # Forward-predict activity from the model
        self.data['predicted'] = -np.diff(self.data['inventory'],axis=0)/\
            np.diff(np.vstack((np.zeros((1,self.max_iter)), \
            np.tile(np.expand_dims(self.data['mass'],1), \
            (1,self.max_iter)))),axis=0)

    def cic(self):
        # Fit exponential curves to unsupported Pb-210 activity profile and derive
        # age and rate assuming constant initial concentration. See example:
        # Donnelly and Bertness (2001) PNAS 98: 14218-14223.
        # Sometimes background/supported level of Pb-210 is unknown. Just in case,
        # this algorithm fits an intercept to the data that predicts the background
        # rate. When background samples are present, these intercept values should
        # be near zero
        # Initialize variables
        self.data['initial'] = np.ones(self.max_iter) * np.nan
        self.data['rate'] = np.ones(self.max_iter) * np.nan
        self.data['intercept'] = np.ones(self.max_iter) * np.nan
        self.data['rsquared'] = np.ones(self.max_iter) * np.nan
        # Step through the Monte Carlo iterations
        for ii in np.arange(self.max_iter):
            # Define initial guess at y-intercept
            self.temp_unsupport = np.mean(self.data['unsupported']\
                [self.data['depth']==np.min(self.data['depth']),ii],axis=0)

            # Fit an exponential model to the data, with and without predicting
            # a baseline background
            if self.bkgrd < 0:
                temp_fit = cfit(self.cic_forward,\
                    self.data['mass'],self.data['unsupported'][:,ii], \
                    p0=np.array([self.temp_unsupport,self.initial_depo_rate,\
                    self.initial_depo_intercept]))[0]
                self.data['intercept'][ii] = temp_fit[2]
            else:
                temp_fit = cfit(self.cic_forward_nointercept,\
                    self.data['mass'],self.data['unsupported'][:,ii], \
                    p0=np.array([self.temp_unsupport,self.initial_depo_rate]))[0]
                self.data['intercept'][ii] = 0

            # Extract model parameters
            self.data['initial'][ii] = temp_fit[0]
            self.data['rate'][ii] = temp_fit[1]
            self.data['rsquared'][ii] = np.corrcoef(self.data['unsupported'][:,ii],\
                self.cic_forward(self.data['unsupported'][:,ii],\
                self.data['initial'][ii],self.data['rate'][ii], \
                self.data['intercept'][ii]))[1,0]**2
            self.data['age'][:,ii] = self.data['mass']/self.data['rate'][ii]

        # Forward-predict the activity values
        self.data['predicted'] = self.data['initial'] * \
            np.exp(-self.decay*self.data['age'])

    def cic_forward(self,predictors,C0,deposition_rate,intercept):
        # Forward model used to fit exponential to the data
        return C0 * \
                np.exp(-self.decay*predictors/deposition_rate) + intercept

    def cic_forward_nointercept(self,predictors,C0,deposition_rate):
        # Forward model used to fit exponential to the data
        return C0 * \
                np.exp(-self.decay*predictors/deposition_rate)

    def plot(self,axes):
        # Plot visualizations of the model fit to the data
        if np.size(axes)<1:
            fig,axes = plt.subplots(2,1)

        if self.model == 'cic' and self.bkgrd<0:
            # Plot supported + modeled intercept values
            axes[0].fill_between(self.data['depth'],\
                np.percentile(self.data['supported']+ \
                np.tile(np.transpose(np.expand_dims(self.data['intercept'],1)),\
                (self.num_data,1)),2.5,axis=1),\
                np.percentile(self.data['supported']+ \
                np.tile(np.transpose(np.expand_dims(self.data['intercept'],1)),\
                (self.num_data,1)),97.5,axis=1),\
                color='m',alpha=0.5,label='Supported Pb-210 + intercept')
        # Plot supported values
        axes[0].fill_between(self.data['depth'],\
            np.percentile(self.data['supported'],2.5,axis=1),\
            np.percentile(self.data['supported'],97.5,axis=1),\
            color='g',alpha=0.5,label='Supported Pb-210')
        # Plot total activity with error visible
        axes[0].errorbar(self.data['depth'],self.data['activity'],\
            yerr=self.data['error'],\
            fmt='bs',label='Total Pb-210')
        # Plot the modeled activity values
        axes[0].fill_between(self.data['depth'],\
            np.nanpercentile(self.data['predicted'] + \
                self.data['supported'],2.5,axis=1),\
            np.nanpercentile(self.data['predicted'] + \
                self.data['supported'],97.5,axis=1),\
            color='r',alpha=0.25)
        axes[0].plot(self.data['depth'],np.nanmedian(self.data['predicted'] + \
            self.data['supported'],axis=1),'r-',label='Modeled')

        # Set chartjunk
        axes[0].set_xlabel('Depth (m)')
        axes[0].set_xlim([np.min(np.hstack((np.array([0]),self.data['depth']))),\
                            np.max(self.data['depth'])*1.05])
        axes[0].set_ylabel(r'Activity (Bq kg$^{-1}$)')
        axes[0].legend()

        # In the second subplot, plot age v depth
        axes[1].fill_between(self.data['depth'],\
            np.nanpercentile(self.data['age'],2.5,axis=1),\
            np.nanpercentile(self.data['age'],97.5,axis=1),\
            color='r',alpha=0.5,label='95% CI')
        axes[1].plot(self.data['depth'],np.nanmedian(self.data['age'],axis=1),\
            'k-',label='Median')

        # Set chartjunk
        axes[1].set_xlim([np.min(np.hstack((np.array([0]),self.data['depth']))),\
                            np.max(self.data['depth'])*1.05])
        axes[1].set_ylabel('Cal. yr before recovery')
        axes[1].set_xlabel('Depth (m)')
        if self.model == 'cic':
            axes[1].annotate(r'Dep. rate = {:.4f} $\pm$ {:.5f} kg/m$^2$/yr'.format(\
                np.nanmedian(self.data['rate']),np.nanstd(self.data['rate'])),\
                [0.1,0.85],xycoords='axes fraction')
        axes[1].legend(loc='lower right')

        plt.show()


    def log_cumtrapz(self,activity,depth):
        # Performs numerical integration assuming exponential decay between
        # depths and not linear as with normal scint.cumtrapz
        # Takes Pb210 activity and cumulative dry weight-depth as input
        output = np.zeros(np.shape(activity))
        output[1,:] = output[0,:] + 0.5 * (activity[1]+activity[0]) * \
                      (depth[1,:]-depth[0,:])
        for ii in np.arange(2,np.shape(output)[0]):
            output[ii,:] = output[ii-1,:] + (activity[ii-1,:]-activity[ii,:]) /\
                            np.log(activity[ii-1,:]/activity[ii,:]) *\
                            (depth[ii,:]-depth[ii-1,:])
        return output







