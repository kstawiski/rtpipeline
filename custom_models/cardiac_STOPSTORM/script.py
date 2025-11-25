#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Luuk van der Pol [UMC Utrecht]: l.h.g.vanderpol@umcutrecht.nl
Please acknowledge our work when using this code.

Networks can be found at: https://zenodo.org/records/13752275
The publication on the development and testing of this work can be found at: https://doi.org/10.1016/j.radonc.2024.110610

Pipeline for running nnUNetv2 with the STOPSTORM trained networks for auto-contouring cardiac substructures.
"""

# All of the following modules should be installed in order to run this script.
import subprocess 
import SimpleITK as sitk
import numpy as np
import glob
import os
from platipy.dicom.io.nifti_to_rtstruct import convert_nifti
import copy


InputPath = "/local_scratch/cardiacdl/StopstormBMcontours/SeparateTest/Input"
OutputPath = "/local_scratch/cardiacdl/StopstormBMcontours/SeparateTest/Output"

Networks = ["603", "605"]
# 504: Lung only (large structures)
# 604: VT only (large structures)
# 603: VT only (small structures)
# 605: Lung + VT (large structures)

CombineNetworks = ["603", "605"] # These networks will be combined into one dicom RT struct. 
# Note, another RT struct will be created for every instance in Networks that is not in this list.
# Note2, if structures are the same for entries. The one from the higher number network will be chosen (this is likely better).


os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':' + os.environ.get('CONDA_PREFIX', '') + '/lib/'
os.environ['nnUNet_raw'] = ' ' # Can stay empty
os.environ['nnUNet_preprocessed'] = ' ' # Can stay empty
os.environ['nnUNet_results'] = '/local_scratch/cardiacdl/nnUNet_results' # Has to be the location of the network.
# ^ make sure to set your local environment variables of nnUNet. The first entry is likely static.
# os.environ['MKL_SERVICE_FORCE_INTEL'] = 'TRUE' # Only required for training networks

# List of structures. Should not change for STOPSTORM networks
ListOfLargeCardiacStructures = ["Atrium_L", "Atrium_R", "Ventricle_L", "Ventricle_R", "V_Venacava_I", "V_Venacava_S", "A_Aorta", "A_Pulmonary"]
ListOfSmallCardiacStructures = ["Valve_Aortic", "Valve_Mitral", "Valve_Pulmonic", "Valve_Tricuspid", "A_LAD", "A_LCX", "A_LM", "A_RCA"]




# Start processing here: No manual adjustments required below this point!
if InputPath[-1]!="/":
    InputPath = InputPath + "/*"
elif InputPath[-1]=="/":
    InputPath = InputPath + "*"
    
if OutputPath[-1]!="/":
    OutputPath = OutputPath + "/"
    
InputPath = glob.glob(InputPath)
InputPath.sort() # Allows easier overview of where the script currently is. Additionally, could run the script in two ways (mind chaning the used GPU)
LIP = len(InputPath)

print(f"Detected {LIP} inputs")

InputCases = []
InputCasesPaths = []
NiftiNames = []
# Write .nii images for all dicom inputs.
for SingleInput in range(LIP):
    CurrentInputPath = InputPath[SingleInput]
    StartSlash = CurrentInputPath.rfind("/")
    CurrentCase = CurrentInputPath[StartSlash+1:]
    InputCases.append(CurrentCase)
    
    # Read dicom images:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(CurrentInputPath)
    InputCasesPaths.append(CurrentInputPath)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Find entry name:
    PositionSlash = CurrentInputPath.rfind("/")
    DatasetName = CurrentInputPath[PositionSlash+1:]
    OutputFolder = OutputPath+DatasetName
    
    # Create dir if not yet present (if present, might be trouble):
    if not os.path.isdir(OutputFolder):
        os.mkdir(OutputFolder)
    NiftiDir = OutputFolder + "/Nifti"
    if not os.path.isdir(NiftiDir):
        os.mkdir(NiftiDir)
    
    # Write nifti image:
    NiftiOutputPath = NiftiDir + "/Image001_0000.nii.gz"
    NiftiNames.append(NiftiOutputPath)
    sitk.WriteImage(image,NiftiOutputPath)

print("Converted inputs to .nii ")

# Maybe create something to get NiftyNames (if already converted before)
# Run nnUNetv2 on the converted data.
NNUNames = []
for Nifty in NiftiNames:
# Could be quicker if all nifties were in 1 folder. However, this would also give 1 output folder that would require sorting.
    
    # Find folder (yes, has to be folder) that contains the nii image:
    PositionSlash = Nifty.rfind("/")
    NiftyInput = Nifty[:PositionSlash+1]
    
    # Define output folder for nnu:
    PositionSlash = NiftyInput.rfind("/")
    OutputFolder = NiftyInput[:PositionSlash]
    
    # Run required networks:
    for OneNNU in Networks:

        OutputNNU = OutputFolder + "Labels" + f"Network_{OneNNU}"
        # SECURITY FIX: Use proper command list without shell=True to prevent command injection
        ActiveCommand = [
            'nnUNetv2_predict',
            '-i', str(NiftyInput),      # Input image
            '-o', str(OutputNNU),       # Output directory (not a single .nii file. Will also contain plan and predictions)
            '-d', str(OneNNU),          # Dataset ID or task number
            '-c', '3d_fullres',         # Specify the model (2d, 3d_lowres, 3d_fullres)
            '-f', 'all',                # Specify which folds to use, e.g., 'all', '0', '1'
        ]

        print(f"Running nnUNet {OneNNU} for {NiftyInput}...")
        result = subprocess.run(ActiveCommand, capture_output=True, text=True)
        # print(result.stdout) # Full output that you would get on commandline. Too much to have here I would say.
        print(f"Completed NNUNET run for: {OutputNNU}")
        NNUNames.append(OutputNNU)
        
        # Print error code if fails.
        if result.returncode != 0:
            print("Error:", result.stderr)


print("Converting outputs to RT structs")
# Convert output to RT struct files:
# Also separate as it could be run individually.
for SingleInput in range(LIP):
    # Same loop length as input thus as inputcases
    CurrentCase = InputCases[SingleInput]
    CurrentOutput = OutputPath + CurrentCase
    
    SeparateStructsPath = CurrentOutput+"/SeparatedStructs"
    if not os.path.isdir(SeparateStructsPath):
        os.mkdir(SeparateStructsPath)
    
    BinarySelectionOutput = []
    for OutputName in range(len(NNUNames)):
        BinarySelectionOutput.append(("/"+CurrentCase+"/") in NNUNames[OutputName]) # (/CurrentCase/) to counter problems with short names (E.g., CAUG has 2/3/4/5/6/7. 6 is a problem, as also in network names, thus all outputs are ok)
    BinarySelectionInput = []
    for InputName in range(len(InputCasesPaths)):
        BinarySelectionInput.append(("/"+CurrentCase) in InputCasesPaths[InputName])
        
    DicomImagePath = InputCasesPaths[BinarySelectionInput.index(True)]
    dcm_file = glob.glob(DicomImagePath + "/*.dcm")
    if len(dcm_file)==0:
        dcm_file = glob.glob(DicomImagePath + "/*")
    dcm_file = dcm_file[0] # Just need one file, not the whole set
    
    OutcomesTrue = [i for i, x in enumerate(BinarySelectionOutput) if x]
    OutcomePaths = []
    for i in range(len(OutcomesTrue)):
        OutcomePaths.append(NNUNames[OutcomesTrue[i]])
        
    for index, ActiveNetwork in enumerate(Networks):
        NetworkOutputPath = SeparateStructsPath + f"/Structs_{ActiveNetwork}"
        if not os.path.isdir(NetworkOutputPath):
            os.mkdir(NetworkOutputPath)
        
        OutputRTSSpath = CurrentOutput + f"/AutomatedRTSS_{ActiveNetwork}.dcm" 
        
        # Process network contours --> Separate + Create RT struct
        AutoContoursPath = OutcomePaths[index] + "/Image001.nii.gz" # Output path is static, index is same order as networks
        if ActiveNetwork == "603":
            ActiveStructureSet = ListOfSmallCardiacStructures
        else:
            ActiveStructureSet = ListOfLargeCardiacStructures
        
        ACstructures = sitk.ReadImage(AutoContoursPath)
        SA = sitk.GetArrayFromImage(ACstructures)
        ImSpacing = ACstructures.GetSpacing()
        
        

        print(f"Starting structure separation for network: {ActiveNetwork}")
        for Label in range(np.max(SA)+1): # 1-8, both for small and for large
            # Get and split all the small structures
            SeparateStructure = copy.deepcopy(SA)
            SeparateStructure[SeparateStructure != Label] = 0
                    
            ActiveStructure = ActiveStructureSet[Label - 1]
            SaveName = NetworkOutputPath + "/Structure_" + ActiveStructure + "_AUTO.nii.gz"
                
            SSI = sitk.GetImageFromArray(SeparateStructure)
            SSI.SetSpacing(ImSpacing)
            sitk.WriteImage(SSI, SaveName)
            
        # Create dicom RT struct
        StructureList = glob.glob(NetworkOutputPath+"/**")
        StructureList.sort() # Don't think its necessary though
        mask = {}
            
        for Struct in range(len(StructureList)):
            CurrentStruct = StructureList[Struct]
            start = CurrentStruct.find("Structure_") + 10 # 10 for the length of Structure_
            end = CurrentStruct.find("_AUTO")
            MaskName = CurrentStruct[start:end]
            #MaskImage = sitk.ReadImage(CurrentStruct)
            mask[f"{MaskName}"] = CurrentStruct
            # print(f"{MaskName} included in the mask dictionary")
        
        print(f"Now creating dicom RT struct for network: {ActiveNetwork}")
        convert_nifti(dcm_file, mask, OutputRTSSpath)
            
    # Now for the combined set
    NetworkNumbers = [int(i) for i in CombineNetworks]
    if 603 in NetworkNumbers:
        Index603 = [i for i, x in enumerate(NetworkNumbers) if x==603]
        Numbers = copy.deepcopy(NetworkNumbers)
        Numbers.pop(Index603[0]) # Should only be 1 entry of 603
        Highest = np.max(Numbers) # 605>604>504. Although, maybe 604 would be better.
        Highest = [603, Highest]
            
    else: # Pick highest.
        Highest = np.max(NetworkNumbers)
        Highest = [Highest]
        print(f"No addative network runs detected. 603 (small structures) missing. This will produce {Highest} as main output")

    # Create combined RT struct.
        # Combine the following networks:
    CN = Highest
    mask = {}
    
    for ActiveNetwork in CN:
        ANS = str(ActiveNetwork)
        NetworkOutputPath = SeparateStructsPath + f"/Structs_{ANS}"
        StructureList = glob.glob(NetworkOutputPath+"/**")
        StructureList.sort() # Don't think its necessary though
            
        for Struct in range(len(StructureList)):
            CurrentStruct = StructureList[Struct]
            start = CurrentStruct.find("Structure_") + 10 # 10 for the length of Structure_
            end = CurrentStruct.find("_AUTO")
            MaskName = CurrentStruct[start:end]
            #MaskImage = sitk.ReadImage(CurrentStruct)
            mask[f"{MaskName}"] = CurrentStruct
            # print(f"{MaskName} included in the mask dictionary")
    
    if len(CombineNetworks)!=0: # Otherwise it makes no sens
        print("Now creating combined output RT struct")
        OutputRTSSpath = CurrentOutput + "/AutomatedRTSS.dcm" 
        convert_nifti(dcm_file, mask, OutputRTSSpath)


#command = [
#    'nnUNetv2_predict',
#    '-i', '/path/to/input/images',      # Input images directory
#    '-o', '/path/to/output/directory',  # Output directory for predictions
#    '-d', 'dataset_id',                 # Dataset ID or task number
#    '-c', 'nnUNetTrainer',              # Specify the trainer class
#    '-t', 'nnUNetTrainerV2',            # The trainer you want to use
#    '--model', '3d_fullres',            # Specify the model (2d, 3d_lowres, 3d_fullres)
#    '--folds', 'all',                   # Specify which folds to use, e.g., 'all', '0', '1'
#    '--num_threads_preprocessing', '8', # Example option for setting preprocessing threads
#    '--num_threads_nifti_save', '4'     # Example option for setting nifti save threads
#]
