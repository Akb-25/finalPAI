############### PAI configuration file ###############
import torch 
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#A global pointer to the tracking class
pbTracker = []


# A flag that you have already reviewed the unwrapped modules
unwrappedModulesConfirmed = False
# A flag that you are doing initial testing and dont need to do a full run
testingDendriteCapacity = False
# A flag which can be set to false use torch.save instead of safetensors
usingSafeTensors = True

# List of nn.Module's which should be converted to Dendrite blocks.
# Anything in this list will receive a duplicate which learns with Perforated Backpropagation
modulesToConvert = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]
# Same as above but add them by name such as "Conv1d" in this list
moduleNamesToConvert = ['PBSequential']
# All Modules should either be converted or tracked
# Tracked modeules just correctly change to stop learning during p mode
# These are checked after modules to Convert so can contain the same things
modulesToTrack = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]
moduleNamesToTrack = []
# This list can have modules added to if if you specifically want to not convert any of the modules in modulesToConvert that are submodles.
modulestoSkip = []
#Same as above but for modules by their ID name which can be copied directly from the debgging statements that are printed.
moduleNamesToSkip = []



'''
Replacement modules happen before the conversion, so replaced modules will then also be run 
through the conversion steps. This can be used if you have a pretrained model but need to
change the way some of the modules are set up.  Fill these in with a pointer to the module 
and the replacement class which takes the initial module as a parameter.  See customization
section 4 for more information.
'''
modulesToReplace = []
replacementModules = []

#If your modules require processing functions add them to this list
modulesWithProcessing = []
#Then add the processing classes to this list
moduleProcessingClasses = []
#Same as above put can pass the modules in by name
moduleNamesWithProcessing = []
moduleByNameProcessingClasses = []

'''
inputDimensions needs to be set every time. It is set to what format of tensors you are
expecting.  Node index should be set to 0, other indexes should be set to -1.  For
example, if your format is [batchsize, nodes, x, y] inputDimensions is [-1,0,-1-1].
if your format is, [batchsize, timestep, nodes] indexesBeforeNode is [-1,-1,0]
'''
inputDimensions = []

#Constants
'''
Percentage Improvement increase needed to call a new best validation score
If your trainer seems to just go forever making smaller and smaller improvements,
this can be set to a larger value to tell the tracker minor improvements dont count.
If you are minimizing loss rather than maximizing an accuracy sore it is more
likely this will need to be raised.
'''
improvementThreshold = 0.001 
#Raw increase needed, if its lower than this its not really learning anyway
improvementThresholdRaw = 1e-5
#Improvement increase needed to call a new best Perforated Backpropagation score
pbImprovementThreshold = 0.01 
#Raw increase needed, if its lower than this its not really learning 
pbImprovementThresholdRaw = 1e-5

# Switch Mode settings
switchMode = -1
# Switch after every epoch
doingSwitchEveryTime = 0

# Switch after no improvement over a set number of epochs
doingHistory = 1
# Make sure these numbers are higher than the scheduler patience
nEpochsToSwitch = 10  # Number of normal epochs to cause a switch
pEpochsToSwitch = 10  # Number of Perforated Backpropagation epochs to cause a switch
capAtN = False #Makes sure PB rounds last max as long as normal rounds
# Number of epochs to average validation scores over
# Set to 1 if you don't want to use averaging. Set to higher than one if your scores fluctuate highly
# and you want to choose a average good point rather than a lucky epoch that may have overfit validation
historyLookback = 1
# Initially after switches number of epochs to wait to make sure you have a fair
# initialization score before tracking any new maxes and and allowing switches to happen
# Set to 0 if you do not want to do averaging.
initialHistoryAfterSwitches = 0

# Switch after a fixed number of epochs
doingFixedSwitch = 2
fixedSwitchNum = 250
#You can set the first switch to be longer than the others for a slower initialization.
#It will not go shorter, so set this lower than fixedSwitchNum to ignore
firstFixedSwitchNum = 249 

#This is for if you set doingPB to be false and just want to run without PB learning but 
#generate the same graphs and csvs
doingNoSwitch = 3

# When creating new tensors for buffers a type must be specified.  Change this value if you are not using floats.
dType = torch.float

# Typically PB nodes will be deleted if the normal learning doesn't improve after adding them.
# This will retain them anyway, generally only used when testing your GPU capacity and running
# A bunch of PB cycles in a row
retainAllPB = False

# This will test various learning rates after each PB cycle.  Often a lower initial rate is
# better so the learning doesn't jump away far from the local minimum the Dendrite nodes trained on
findBestLR = True

# Set to 1 if you want to quit as soon as one Dendrite fails
# Higher numbers will try a new set of randomly initialized Dendrites
maxDendriteTries = 5
#this is the maximum number of dendrites that will be created.  Can be set lower to quit early
maxDendrites = 100

# This number is to check how many batches to average out the initial correlation score over
initialCorrelationBatches = 100 #this should be at least 100 and up to 10% of a whole epoch

# This is the forward function used by Dendrites which can be changed
PBForwardFunction = torch.sigmoid

'''
A custom PAI module which can be used to group layers into a single block

This takes in an array of layers. For example:

    PBG.PBSequential([nn.Linear(2 * hidden_dim, seqWidth),
            nn.LayerNorm(seqWidth)])
    
    This should be used for all normalization layers.
    You will get warnings if normalization layers are unwrapped.
    
'''
class PBSequential(nn.Sequential):
        def __init__(self, layerArray):
            super(PBSequential, self).__init__()
            self.model = nn.Sequential(*layerArray)
        def forward(self, x):
            return self.model(x)