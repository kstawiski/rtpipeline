

# **A Comprehensive Guide to DICOM-RT Data Extraction for Varian and Elekta Systems: From Tags to Insights**

## **Part I: The DICOM-RT Ecosystem \- A Foundational Overview**

### **The Clinical Workflow as a Data Narrative**

To construct a robust data extraction pipeline for radiotherapy, it is essential to first understand that the collection of Digital Imaging and Communications in Medicine \- Radiotherapy (DICOM-RT) files for a single patient is not merely a set of disconnected data points. Instead, these files form an interconnected, chronological data narrative that digitally mirrors the entire clinical treatment process.1 Architecting a pipeline in alignment with this narrative structure is fundamental to preserving the relational integrity and contextual meaning of the data.

The narrative begins with diagnostic and simulation imaging. A patient is scanned on a Computed Tomography (CT) scanner, and potentially with Magnetic Resonance (MR) or Positron Emission Tomography (PET) scanners, to produce a set of standard DICOM images that visualize the patient's anatomy.1 These images serve as the foundational canvas upon which the entire treatment is designed.

Following imaging, a radiation oncologist or dosimetrist performs a process known as virtual simulation and contouring. During this phase, they identify and delineate critical anatomical structures on the planning images. These include the tumor volumes to be treated (targets) and the healthy organs to be avoided (organs at risk, or OARs). This process generates the first radiotherapy-specific object: the **RT Structure Set**.1

With the anatomical map established, the treatment planning phase commences. A medical physicist or dosimetrist uses a Treatment Planning System (TPS) to design the radiation delivery. This involves selecting beam energies, angles of approach, and shaping the beams to conform to the target while sparing the OARs. This highly complex process results in two critical outputs: the **RT Plan**, which contains the complete set of instructions for the treatment machine, and the **RT Dose**, a three-dimensional matrix representing the calculated radiation dose distribution that would result from delivering the plan.1

Finally, the patient undergoes treatment, which is typically delivered in a series of daily sessions or "fractions." During or after each session, the treatment delivery system (e.g., a linear accelerator) or its associated Record and Verify (R\&V) system records the actual machine parameters that were used. This generates a historical log for each session, known as an **RT Treatment Record**.1 These records are vital for quality assurance, allowing for a direct comparison between the intended plan and the actual delivery.

This workflow—from imaging to planning to delivery and verification—is the logical framework that dictates the purpose and interdependencies of the DICOM-RT objects. A data pipeline that fails to model these relationships will yield a collection of isolated files, whereas a pipeline that respects this narrative structure can construct a comprehensive, relational digital twin of a patient's treatment course.

### **Core DICOM-RT Information Object Definitions (IODs)**

The DICOM standard, originally developed for radiology, was extended in 1997 to include objects specific to radiotherapy, with further additions in 1999\.1 These Information Object Definitions (IODs) standardize the structure for storing and transmitting the complex data required for radiation treatment. Understanding the distinct role of each primary IOD is the first step in targeted data extraction.

* **RT Structure Set (RTSTRUCT):** This object defines the 'where' of the treatment. It contains the geometric definitions of all relevant anatomical structures, known as Regions of Interest (ROIs). These are typically stored as a series of 2D contours drawn on the axial slices of the planning image set.5 The RTSTRUCT file is the anatomical blueprint for the entire treatment. Its  
  Modality (0008,0060) tag is RTSTRUCT.6  
* **RT Plan (RTPLAN):** This object defines the 'how' of the treatment. It is one of the most complex IODs in the DICOM standard, containing the complete geometric and dosimetric specification for a course of therapy.1 It details every parameter required to deliver the treatment, including beam energies, gantry and collimator angles, beam modifiers like wedges, and, for modern techniques, the precise position of each multi-leaf collimator (MLC) leaf at hundreds of control points. It also contains the prescription information, such as the total dose and number of fractions. Its  
  Modality (0008,0060) tag is RTPLAN.7  
* **RT Dose (RTDOSE):** This object defines the 'what'—the predicted outcome of the plan. It stores the calculated spatial distribution of radiation dose as a three-dimensional grid of voxels, where each voxel has a value corresponding to the absorbed dose.10 This "dose cube" can be overlaid on the patient's images to visualize how the dose conforms to the target and spares healthy tissues. Its  
  Modality (0008,0060) tag is RTDOSE.7  
* **RT Beams Treatment Record:** This object, a specific type of RT Treatment Record, defines 'what happened'. It is a historical log of an actual treatment session delivered via external beams.4 It contains the machine parameters as they were recorded during delivery, such as the actual monitor units delivered and the gantry angles traversed. This allows for a direct comparison against the planned parameters in the RTPLAN file, forming the basis of delivery quality assurance. Its  
  Modality (0008,0060) tag is RTRECORD.7

### **The Web of Data: Linking Objects with Unique Identifiers (UIDs)**

The relational integrity of the radiotherapy data narrative is maintained through a robust system of Unique Identifiers (UIDs), a cornerstone of the DICOM standard.13 An extraction pipeline must not only parse the content of individual files but also parse these UIDs to reconstruct the critical links between them. Failure to do so results in a disconnected dataset where spatial and logical relationships are lost.

* **SOP Instance UID (0008,0018):** This is the universal primary key for any DICOM object. Each file, whether it is a single CT slice, an RTPLAN, or an RTDOSE, has a globally unique SOP Instance UID. This tag is used for direct, unambiguous referencing between objects.  
* **Series Instance UID (0020,000E):** This UID groups a set of related objects. While the DICOM standard specifies that a series should contain objects of only one modality, this is a point of frequent vendor variation, and in practice, a single series may contain multiple RT objects.7 For example, an RTPLAN, its associated RTDOSE, and RTSTRUCT may all share a Series Instance UID, or they may exist in separate series within the same study.  
* **Study Instance UID (0020,000D):** This UID groups all series related to a single clinical event or "study" for a patient. For a single course of radiotherapy, all associated images, structures, plans, doses, and records should share the same Study Instance UID, linking them all to the same treatment endeavor.  
* **Frame of Reference UID (0020,0052):** This is arguably the most critical UID for any form of spatial analysis. It identifies a specific, unique three-dimensional coordinate system. For the geometric data in an RTSTRUCT (contours), an RTPLAN (isocenters, beam geometry), and an RTDOSE (dose grid) to be spatially coherent and validly overlaid, they **must** all share the same Frame of Reference UID. This UID is the lynchpin that guarantees that a point with coordinates  refers to the same physical location in each of these objects. Any analysis that combines spatial information from objects with different Frame of Reference UIDs without an explicit spatial registration transformation is fundamentally invalid.  
* **Referenced SOP Instance UIDs:** The DICOM-RT standard provides explicit mechanisms for linking objects. Many IODs contain sequences that hold the SOP Instance UIDs of other, related objects. For instance, an RTPLAN object contains a ReferencedStructureSetSequence (300C,0060), which includes the SOP Instance UID of the specific RTSTRUCT file upon which the plan is based.14 Similarly, an RTDOSE object contains a  
  ReferencedRTPlanSequence (300C,0002) pointing to the RTPLAN from which the dose was calculated.15 A robust extraction pipeline must be programmed to parse these reference sequences to build a complete and accurate relational graph of the patient's treatment data. The initial step in processing a patient's dataset should be to map out these relationships, using the UIDs as the connecting threads.

## **Part II: Deep Dive into Key DICOM-RT Objects and Tags**

This section provides a granular analysis of the four primary DICOM-RT objects, detailing the essential tags and sequences that a data extraction pipeline must target. For each object, the core purpose, key data structures, and a detailed table of important attributes are presented.

### **Dissecting the RT Structure Set (RTSTRUCT)**

The RT Structure Set's purpose is to define the anatomical Regions of Interest (ROIs) as a collection of 2D contours on the planar slices of an associated image series.6 It provides the geometric context for both planning and evaluation.

The data within an RTSTRUCT object is organized hierarchically. The primary sequences to parse are:

* **StructureSetROISequence (3006,0020):** This is a sequence where each item defines a single ROI. It contains metadata about the ROI, such as its assigned number and, most importantly, its name. The ROIName (3006,0026) tag is of paramount importance for any analysis, but it is notoriously unstandardized, consisting of user-entered free text.6 This necessitates a significant data normalization effort in any large-scale pipeline.  
* **ROIContourSequence (3006,0039):** This sequence links the ROI definitions to their actual geometric data. Each item in this sequence corresponds to an ROI defined in the StructureSetROISequence.  
* **ContourSequence (3006,0040):** Nested within each item of the ROIContourSequence, this sequence contains the individual 2D contour slices that collectively form the 3D ROI.  
* **ContourData (3006,0050):** This tag, within each item of the ContourSequence, holds the raw geometric data. It is a flattened list of floating-point numbers representing the  coordinates of all the vertices that make up that single 2D contour.

The core extraction task for an RTSTRUCT file involves iterating through the StructureSetROISequence to build a dictionary of ROI numbers and names. Then, for each ROI, the pipeline must traverse the corresponding items in the ROIContourSequence and ContourSequence to access the ContourData. This flattened list of coordinates must be reshaped into an array of 3D points. For many research applications, such as radiomics, it is highly beneficial to use this collection of contours to generate a 3D binary mask volume that has the same geometry as the referenced image series. Open-source libraries like rt-utils in Python are specifically designed to automate this complex conversion from contour data to mask volumes.17

**Table 2.1: Essential RTSTRUCT Tags**

| Tag | Name | VR | Description & Clinical/Research Utility |
| :---- | :---- | :---- | :---- |
| (3006,0020) | StructureSetROISequence | SQ | Sequence containing definitions for each ROI. **Utility:** The entry point to all anatomical structures. The primary loop for parsing structures begins here. |
| \>(3006,0022) | ROINumber | IS | Unique integer identifier for the ROI within this Structure Set. **Utility:** Serves as the primary key for linking ROI definitions in StructureSetROISequence to their corresponding geometric data in ROIContourSequence. |
| \>(3006,0026) | ROIName | LO | User-defined name for the ROI (e.g., "PTV", "SpinalCord", "GTV\_L\_LUNG"). **Utility:** The human-readable label for the structure. Critical for all clinical and research analysis but requires significant normalization/mapping due to lack of standardization across institutions and planners. |
| (3006,0039) | ROIContourSequence | SQ | Sequence containing the geometric data for all ROIs, linking them by ReferencedROINumber. |
| \>(3006,0040) | ContourSequence | SQ | Sequence of individual 2D contours that make up a single 3D ROI. |
| \>\>(3006,0050) | ContourData | DS | A flattened list of  coordinates for a single 2D contour, in millimeters. **Utility:** This is the raw geometric data. It is essential for reconstructing 3D surface models or volumetric masks of the anatomy, which are foundational for DVH calculations and radiomics. |
| \>\>(3006,0016) | ContourImageSequence | SQ | Links a contour to a specific image slice via its ReferencedSOPInstanceUID. **Utility:** Confirms which image slice a contour was drawn on, providing an explicit link back to the source image data. |
| (300C,0060) | ReferencedStructureSetSequence | SQ | Sequence referencing the Structure Set this object is based on. |
| \>(0020,0052) | FrameOfReferenceUID | UI | The UID of the coordinate system in which the contour coordinates are defined. **Utility:** Must match the FoR UID of the image, plan, and dose objects for any valid spatial analysis. This tag is the key to spatial integrity. |

### **Deconstructing the RT Plan (RTPLAN)**

The RT Plan is the blueprint of the treatment, describing the entire radiation prescription and the precise technical parameters for its delivery.9 The complexity and monolithic nature of this "first-generation" IOD was a key motivation for the development of the "second-generation" DICOM-RT objects, which aim for a more modular structure.19 For current data, however, parsing this complex object is a necessity.

The plan's data is organized into several critical nested sequences:

* **FractionGroupSequence (300A,0070):** This sequence defines the high-level prescription. A plan may have multiple fraction groups, for example, an initial course followed by a smaller "boost" treatment. Each item in this sequence specifies the NumberOfFractionsPlanned (300A,0078) and the prescribed dose, typically found within the nested DoseReferenceSequence (300A,0080) under the tag TargetPrescriptionDose (300A,0084).  
* **BeamSequence (300A,00B0):** This is the core of the technical plan, containing a sequence of items where each describes a single treatment beam. It includes static parameters like the BeamName (300A,00C2), BeamType (300A,00C4) (e.g., 'STATIC' or 'DYNAMIC'), and the radiation source-to-axis distance.  
* **ControlPointSequence (300A,0111):** For modern modulated therapies like IMRT and VMAT, this is the most important and data-rich sequence. Nested within each beam, this sequence defines a series of "snapshots" of the treatment machine's state. For a simple static beam, there may be only two control points (beam on and beam off). For a rotational VMAT arc, there can be hundreds of control points. Each control point specifies the dynamic machine parameters at that instant.

The core extraction task for an RTPLAN is hierarchical. First, parse the FractionGroupSequence to extract the overall prescription details (total dose, number of fractions). Next, iterate through each item in the BeamSequence. For each beam, extract its static properties. The most intensive part of the process is then to iterate through every control point within that beam's ControlPointSequence. At each control point, the pipeline must extract key dynamic parameters, including GantryAngle (300A,011E), BeamLimitingDeviceAngle (300A,0120) (collimator rotation), CumulativeMetersetWeight (300A,0134), and, critically, the jaw and MLC leaf positions from the BeamLimitingDevicePositionSequence (300A,011A). This granular control point data is essential for understanding the complexity of the plan and for any independent dose calculation or delivery simulation.

**Table 2.2: Essential RTPLAN Tags**

| Sequence/Tag | Name | VR | Description & Clinical/Research Utility |
| :---- | :---- | :---- | :---- |
| (300A,0070) | FractionGroupSequence | SQ | Defines the prescription for a group of fractions (e.g., initial treatment, boost). |
| \>(300A,0078) | NumberOfFractionsPlanned | IS | Total number of sessions for this group. **Utility:** Core prescription data, essential for dose-response modeling and calculating biologically effective doses. |
| \>(300A,0080) | DoseReferenceSequence | SQ | Specifies the target dose for a reference point or volume. |
| \>\>(300A,0084) | TargetPrescriptionDose | DS | The prescribed dose in Gray (Gy) for this fraction group. **Utility:** Core prescription data, representing the intended therapeutic dose. |
| (300A,00B0) | BeamSequence | SQ | Sequence containing definitions for each treatment beam. |
| \>(300A,00C0) | BeamNumber | IS | A unique integer identifier for the beam within the plan. |
| \>(300A,00C2) | BeamName | LO | User-defined name for the beam (e.g., "Gantry 180", "Ant"). |
| \>(300A,00C4) | BeamType | CS | e.g., 'STATIC', 'DYNAMIC'. **Utility:** Distinguishes between simple conformal fields and complex modulated beams (IMRT/VMAT), a key categorical feature for analysis. |
| \>(300A,0111) | ControlPointSequence | SQ | Sequence of machine states that define the beam delivery. This is the heart of IMRT and VMAT plans. |
| \>\>(300A,011E) | GantryAngle | DS | Rotational angle of the treatment machine gantry in degrees. **Utility:** A key geometric parameter defining the beam's direction of entry. |
| \>\>(300A,0120) | BeamLimitingDeviceAngle | DS | Rotational angle of the collimator in degrees. **Utility:** A key geometric parameter defining the orientation of the MLCs and jaws. |
| \>\>(300A,011A) | BeamLimitingDevicePositionSequence | SQ | Sequence defining the positions of beam-shaping devices (jaws and MLCs). |
| \>\>\>(300A,011C) | LeafJawPositions | DS | A flattened list of paired numbers defining the positions of each MLC leaf pair or jaw in millimeters. **Utility:** Defines the beam aperture shape at each control point. This is the raw data for calculating plan complexity metrics and for any independent dose verification. |
| \>\>(300A,0134) | CumulativeMetersetWeight | DS | The fraction of the total beam monitor units (dose) delivered up to this control point (ranges from 0 to 1). **Utility:** Defines the fluence delivered between control points. The difference in weight between two consecutive points determines the dose delivered during that segment. |

### **Analyzing the RT Dose (RTDOSE)**

The RT Dose object's purpose is to store the 3D dose distribution as calculated by the TPS.10 It is essentially a 3D image where voxel intensity represents absorbed dose.

Correctly interpreting this object requires more than just reading the pixel data. Several key tags must be used in conjunction to reconstruct the dose grid accurately in terms of both value and spatial location:

* **PixelData (7FE0,0010):** This tag contains the raw, unscaled 3D dose grid, stored as a long one-dimensional array of integer values. These are not the final dose values.  
* **DoseGridScaling (3004,000E):** This is a mandatory (Type 1C) floating-point number that serves as a global multiplier. The actual dose value for each voxel is obtained by multiplying the integer value from PixelData by this scaling factor.15 This is a frequent point of error in naive extraction pipelines.  
* **Grid Dimensions:** The tags Rows (0028,0010), Columns (0028,0011), and NumberOfFrames (0028,0008) define the dimensions of the 3D grid, allowing the 1D PixelData array to be correctly reshaped.  
* **Spatial Orientation:** A set of tags defines the grid's position and orientation in the patient's coordinate system. ImagePositionPatient (0020,0032) gives the  coordinates of the center of the top-left voxel of the first slice. PixelSpacing (0028,0030) provides the in-plane resolution (x and y spacing). Critically, GridFrameOffsetVector (3004,000C) provides a list of z-offsets from the first slice for every subsequent slice, defining the inter-slice spacing, which may not be uniform.15

The core extraction task for an RTDOSE file is to perform this reconstruction. The pipeline must first read the grid dimensions, the DoseGridScaling factor, and all spatial orientation tags. It must then read the PixelData array, reshape it into a 3D array (e.g., a NumPy array) of size (NumberOfFrames, Rows, Columns), and then perform an element-wise multiplication of this entire array by the DoseGridScaling factor. The result is a 3D matrix where each element is the dose in the units specified by DoseUnits (3004,0002), typically Gray (Gy).15

**Table 2.3: Essential RTDOSE Tags**

| Tag | Name | VR | Description & Clinical/Research Utility |  |
| :---- | :---- | :---- | :---- | :---- |
| (7FE0,0010) | PixelData | OW/OB | The raw 3D dose grid stored as a 1D array of integer pixel values. This is the raw data that must be scaled and reshaped. |  |
| (3004,000E) | DoseGridScaling | DS | The scaling factor to convert pixel values to dose units. **Utility:** Absolutely mandatory for correct dose interpretation. The formula is: . |  |
| (3004,0002) | DoseUnits | CS | Units of dose, typically 'GY' (Gray) or 'RELATIVE'.11 | **Utility:** Confirms the unit of the scaled dose values, essential for all quantitative analysis. |
| (3004,000A) | DoseSummationType | CS | Type of dose summation (e.g., 'PLAN', 'FRACTION', 'BEAM').11 | **Utility:** Informs whether the dose represents a single fraction, a single beam, or the entire multi-fraction plan. Critical for correct interpretation in research. |
| (0028,0010) | Rows | US | Number of rows (y-dimension) in the dose grid. Used to reshape the PixelData array. |  |
| (0028,0011) | Columns | US | Number of columns (x-dimension) in the dose grid. Used to reshape the PixelData array. |  |
| (0028,0008) | NumberOfFrames | IS | Number of slices (z-dimension) in the dose grid. Used to reshape the PixelData array. |  |
| (0020,0032) | ImagePositionPatient | DS | The  coordinates of the center of the top-left corner voxel of the first slice. **Utility:** Anchors the dose grid in the patient coordinate system defined by the Frame of Reference. |  |
| (0028,0030) | PixelSpacing | DS | The physical distance between the centers of adjacent voxels in the x and y directions, in millimeters. |  |
| (3004,000C) | GridFrameOffsetVector | DS | A vector of z-offsets for each slice relative to the first slice, defining the slice spacing. **Utility:** Critical for correct z-dimension scaling; the spacing may not be uniform, especially in older plans. |  |
| (300C,0002) | ReferencedRTPlanSequence | SQ | Sequence referencing the RT Plan that this dose was calculated for.15 | **Utility:** Provides the crucial link from the calculated dose back to the plan that created it. |

### **Interpreting the RT Beams Treatment Record (RTRECORD)**

The RT Beams Treatment Record provides a detailed log of one or more treatment sessions, documenting the machine parameters as they were actually delivered by the linear accelerator.4 Its primary function is for quality assurance, enabling a direct comparison between the planned treatment and the delivered treatment.

The structure of the RTRECORD mirrors that of the RTPLAN, facilitating this comparison:

* **ReferencedRTPlanSequence (300C,0002):** This is the most important linking sequence, containing the SOP Instance UID of the RTPLAN that was intended for delivery. This link is the foundation of any plan-versus-delivered analysis.  
* **TreatmentSessionBeamSequence (3008,0021):** This is the main sequence, containing an item for each beam that was delivered during the recorded session. Each item is linked to its corresponding beam in the RTPLAN via the ReferencedBeamNumber (300C,0004).  
* **ControlPointDeliverySequence (3008,0041):** Similar to the sequence in the RTPLAN, this contains the delivered machine parameters. For dynamic deliveries, it records the actual gantry angles, leaf positions, and meterset values at various points during the beam delivery.12

The core extraction task for an RTRECORD involves leveraging its link to the corresponding RTPLAN. After identifying the referenced plan, the pipeline should iterate through each beam in the TreatmentSessionBeamSequence. For each beam, it should extract key delivered parameters, most notably the DeliveredMeterset (3008,0042). This value can then be compared to the BeamMeterset (300A,0086) from the corresponding beam in the RTPLAN. For dynamic beams, a more granular comparison can be performed by iterating through the ControlPointDeliverySequence and comparing the delivered parameters (gantry angle, MLC positions) to their planned counterparts in the RTPLAN's ControlPointSequence. These comparisons are the basis for automated delivery quality assurance (DQA) and are a rich source of data for research into treatment accuracy and its clinical impact.

**Table 2.4: Essential RT Beams Treatment Record Tags**

| Sequence/Tag | Name | VR | Description & Clinical/Research Utility |
| :---- | :---- | :---- | :---- |
| (300C,0002) | ReferencedRTPlanSequence | SQ | Links this record to the intended RT Plan. **Utility:** The primary link for any plan vs. delivered comparison. Essential for QA and research on delivery accuracy. |
| (3008,0020) | TreatmentDate | DA | Date the treatment session occurred. **Utility:** Provides the temporal context for the treatment, crucial for longitudinal studies. |
| (3008,0021) | TreatmentTime | TM | Time the treatment session occurred. |
| (3008,0021) | TreatmentSessionBeamSequence | SQ | Sequence containing records for each beam delivered in the session.12 |
| \>(300C,0004) | ReferencedBeamNumber | IS | The number of the beam in the referenced RT Plan, linking the delivered beam record to the planned beam. |
| \>(3008,0041) | ControlPointDeliverySequence | SQ | Sequence of delivered machine states for the beam, containing the actual measured parameters during delivery.12 |
| \>\>(300A,011E) | GantryAngle | DS | Actual delivered gantry angle at a control point. **Utility:** Can be compared to the planned angle to check for rotational accuracy. |
| \>\>(300A,011C) | LeafJawPositions | DS | Actual delivered leaf/jaw positions. **Utility:** Can be compared to planned positions to verify geometric accuracy of the aperture. |
| \>(3008,0042) | DeliveredMeterset | DS | The total monitor units delivered for this beam in this session. **Utility:** A key value for verifying correct dose output. A significant deviation from the planned meterset is a critical delivery error. |
| \>(3008,0116) | ApplicationSetupCheck | CS | Indicates if setup was verified (e.g., 'VERIFIED', 'VERIFIED\_OVR' for override). **Utility:** Records whether pre-treatment imaging and positioning checks were passed, providing context on the patient setup accuracy for that session. |

## **Part III: Navigating Vendor-Specific Implementations: Varian and Elekta**

While the DICOM standard provides a blueprint for data exchange, it is crucial to recognize that adherence to the standard does not guarantee uniform implementation across different manufacturers. In practice, vendors may populate optional tags differently, use private (non-standard) tags to store proprietary information, or interpret parts of the standard in unique ways. A successful data extraction pipeline must be robust to these variations. This section focuses on the specifics of the two dominant vendors in radiation oncology: Varian Medical Systems and Elekta.

### **Decoding DICOM Conformance Statements**

The primary tool for understanding a specific product's DICOM capabilities is its Conformance Statement. This is a technical document provided by the manufacturer that explicitly details which parts of the DICOM standard the product supports.2 These statements are the essential starting point for assessing potential interoperability between systems.22

When analyzing a conformance statement, the key sections to examine are:

* **Supported SOP Classes:** This section, often presented in a table, lists all the DICOM objects the device can create (send) or accept (receive). It will specify the role for each object: Service Class User (SCU), which initiates a service like sending a file, or Service Class Provider (SCP), which provides a service like receiving a file.24 For example, a TPS is typically an SCP for CT images (it receives them) and an SCU for RT Plans (it sends them).  
* **Information Object Definition (IOD) Details:** The statement should provide detailed tables specifying which modules and attributes of a given IOD are created or processed. This is where one can find information on how specific tags are populated or if certain optional tags are used.  
* **Private Data Elements:** A dedicated section should list any private tags the device uses, along with their tag numbers, value representations (VRs), and a description of their purpose.

It is critical, however, to understand the "interoperability paradox." Conformance statements almost universally include disclaimers clarifying that they guarantee "inter-connectivity" (the ability to establish a connection and exchange messages) but not "interoperability" (the ability for the receiving application to process the information meaningfully).1 True interoperability can only be confirmed through extensive testing. Nevertheless, these documents are the indispensable first step in identifying potential data extraction challenges. Conformance statements for Varian and Elekta products are available on their respective corporate websites.22 For example, the statement for Varian System Server v18.1 covers the ARIA RTM and Eclipse TPS products 28, while Elekta provides separate statements for products like MOSAIQ and Monaco across various versions.24

### **Varian (ARIA® / Eclipse™) Specifics**

Varian's oncology ecosystem is characterized by the tight integration between its ARIA® Oncology Information System (OIS) and the Eclipse™ Treatment Planning System.28 Data often flows between these systems via a shared database, with DICOM serving as the primary mechanism for importing data from and exporting data to third-party systems.

A key characteristic of Varian's DICOM implementation is the use of private tags to store information that is either proprietary or not well-represented in the standard DICOM-RT objects. The official DICOM Conformance Statement for the Varian System Server is the authoritative source for these tags. For example, some conformance statements note that private attribute values are reserved in the group number  for interfacing with their OIS.33 Extracting this information can provide deeper insights into the treatment plan and its history within the Varian ecosystem.

However, for truly comprehensive access to Varian's planning data, a DICOM-only approach may be insufficient. Varian provides the Eclipse Scripting API (ESAPI), a powerful.NET library that allows for direct, programmatic access to the Eclipse planning database.35 ESAPI can retrieve a wealth of information—such as optimization objectives, clinical goals, and plan uncertainty parameters—that is often not exported or is only partially represented in the DICOM RTPLAN file.36 This suggests a "dual-API" reality for Varian data: DICOM is the standard for interoperability, but ESAPI is the key to deep, native data access. For developers working in Python, the open-source PyESAPI library provides a wrapper that makes it possible to call ESAPI functions from a Python environment, enabling a hybrid data extraction strategy.35

The implementation details specified in conformance statements are highly version-specific. A pipeline developed for data from Varian System Server v16.1 may not function correctly with data from v18.1 without modification. This is not a one-time development challenge but an ongoing maintenance requirement. As clinical sites upgrade their software, the data pipeline must be re-validated and potentially updated to handle changes in how standard tags are used or the introduction of new private tags. This underscores the need for a version-aware architecture and a robust, automated testing framework for any long-term data extraction project.

**Table 3.1: Known Varian Private Tags (Example)**

| Tag | Private Creator | VR | Description & Inferred Utility |
| :---- | :---- | :---- | :---- |
| (32xx,xxxx) | VARIAN | various | Placeholder for various private tags related to the OIS interface. The exact element and meaning must be cross-referenced with the specific Conformance Statement. For example, a tag in this group could store an internal ARIA plan ID or a specific workflow status.33 |
| (300B,1012) | VMS\_6.0 | LO | Beam Generation Mode. May contain values like 'STATIC', 'DYNAMIC', 'ARC', which can be more specific than the standard BeamType tag, potentially distinguishing between different types of arc therapies. |
| (300B,1013) | VMS\_6.0 | DS | Fluence Map. Used in electronic compensation or advanced IMRT planning techniques, this tag likely contains data related to the intensity map of the beam. |

### **Elekta (MOSAIQ® / Monaco®) Specifics**

Elekta's ecosystem is centered around the MOSAIQ® Oncology Information System and the Monaco® Treatment Planning System.37 Like Varian, Elekta provides a suite of DICOM Conformance Statements for their various products and software versions, which are the primary source for understanding their specific implementations.24

Elekta systems also utilize private tags to store information specific to their hardware and software workflows. Public documentation of these tags can be sparse, often requiring empirical analysis of DICOM files exported from Elekta systems. A tool like pydicom is invaluable for this task, as it can parse an entire DICOM file and list all tags, including private ones. The key is to first identify the PrivateCreatorID tag (e.g., (7053, 0010\) might contain 'Elekta Impac'), which then provides the context for interpreting the subsequent private tags within that group (e.g., (7053, 10xx)).41

In recent years, Elekta has focused on improving interoperability, particularly between MOSAIQ and Monaco. Newer versions feature capabilities like automated DICOM-RT plan export and improved management of Spatial Registration Objects, which can simplify data transfer workflows.40 Furthermore, Elekta has begun to offer alternative data access routes beyond DICOM. The Elekta FHIR API for MOSAIQ allows developers to retrieve patient data using modern web standards (FHIR, REST), which may be a more streamlined approach for accessing demographic or clinical summary data than parsing it from DICOM headers.43

While private tags can be a challenge, they can also contain valuable data. For example, a private tag might hold an internal database identifier that could be used to link the DICOM object back to a record in a research database or a clinical trial management system. Therefore, the goal should not be to simply discard private tags, but to identify them, understand their purpose through documentation or empirical study, and selectively extract them when they provide meaningful information. This approach, however, must be balanced with the need for data anonymization, as private tags have no guarantee of being free of Protected Health Information (PHI).

**Table 3.2: Known Elekta Private Tags (Example)**

| Tag | Private Creator | VR | Description & Inferred Utility |
| :---- | :---- | :---- | :---- |
| (7053,xx\_xx) | Elekta Impac | various | Group used by MOSAIQ for various internal parameters. The specific tags are often related to workflow states, internal database IDs, or site-specific configurations. Extracting these may be useful for operational research but requires careful, site-specific validation. |
| (300D,xx\_xx) | IMPAC\_RTPLAN\_1 | various | Private group found in RT Plan objects created by older IMPAC/Elekta systems. May contain legacy parameters related to dose calculation or machine models that are not part of the standard DICOM object. |

## **Part IV: A Practical Guide for Data Extraction and AI Agent Instruction**

This section transitions from theoretical understanding to practical implementation. It provides a technical blueprint for building the extraction pipeline and, crucially, a detailed instruction set for translating the extracted raw data into the meaningful clinical and research insights required by an AI agent.

### **Technical Blueprint for the Extraction Pipeline**

Python is the language of choice for medical imaging research and data science due to its extensive ecosystem of powerful, open-source libraries. A robust DICOM-RT extraction pipeline can be built upon the following stack:

* **pydicom:** This is the foundational library for all DICOM-related tasks in Python. It provides the core functionality to read, parse, modify, and write DICOM files.44 Its strength lies in its ability to handle the full complexity of the DICOM standard, including nested sequences and private data elements.41  
* **rt-utils / pydicom-rt:** These are higher-level libraries built on top of pydicom. They are specifically designed to simplify common radiotherapy tasks. For example, rt-utils provides functions to easily convert the complex contour data from an RTSTRUCT file into a 3D NumPy array representing a binary segmentation mask, a format far more amenable to machine learning applications.17  
* **NumPy:** This library is essential for any numerical computation. It is used to efficiently handle the 3D dose grids from RTDOSE files, the arrays of coordinates from RTSTRUCT files, and the binary masks generated from them.  
* **pandas:** This library is ideal for organizing the extracted metadata. A common workflow is to parse the DICOM tags from a cohort of patients and store them in a pandas DataFrame, where each row represents a patient or a specific DICOM object, and each column represents a DICOM tag or a derived parameter.

**Parsing Private Tags with pydicom**

A key task for a comprehensive pipeline is the handling of vendor-specific private tags. The DICOM standard reserves tag groups with odd numbers for private use. To interpret these tags, one must first find the "Private Creator" tag, which is located at an element between (gggg, 0010\) and (gggg, 00FF), where gggg is the odd group number. This creator tag contains a string that identifies the vendor or implementation. This string then provides the context for all other tags in that group, which are located at elements (gggg, xx00) through (gggg, xxFF), where xx corresponds to the creator tag's element.41

The following Python code demonstrates how to explore and access private tags using pydicom:

Python

import pydicom

\# Load a DICOM file that may contain private tags  
\# ds \= pydicom.dcmread("path/to/varian\_or\_elekta\_plan.dcm")

\# To discover private creators, iterate through the dataset  
private\_creators \= {}  
for elem in ds:  
    if elem.tag.is\_private:  
        \# The private creator tag has an element number from 0x10 to 0xFF  
        if 0x0010 \<= elem.tag.element \<= 0x00FF:  
            if elem.VR \== 'LO': \# Private creators are of VR 'LO'  
                private\_creators\[elem.tag.group\] \= elem.value

\# print("Found Private Creators:", private\_creators)  
\# Example output might be: {0x300b: 'VMS\_6.0', 0x7053: 'Elekta Impac'}

\# Once a creator is known, you can access its private tags.  
\# This requires knowing the specific tag element from the conformance statement or by inspection.  
\# For example, if we know Varian uses (300B, 1012\) for 'Beam Generation Mode'  
varian\_creator\_group \= 0x300B  
beam\_gen\_mode\_element \= 0x1012  
private\_tag \= pydicom.tag.Tag(varian\_creator\_group, beam\_gen\_mode\_element)

\# if private\_tag in ds:  
\#     beam\_mode \= ds\[private\_tag\].value  
\#     print(f"Varian Private Beam Generation Mode: {beam\_mode}")

### **Instruction Set for AI Agents: Translating Tags into Actionable Insights**

The ultimate goal of the extraction pipeline is to produce data that is not just stored but understood. An AI agent tasked with analyzing this data needs to know the clinical and research significance of each parameter. The following table serves as a "Rosetta Stone," translating key clinical concepts into specific DICOM tags and providing instructions for their interpretation and use. This process of mapping and standardizing raw data is a form of data harmonization, which is essential for building reliable models from multi-vendor, multi-institutional datasets. The AAPM's Task Group 263 report on standardizing nomenclature provides an excellent framework for the kind of harmonization required, particularly for ROI names.46

**Table 4.1: AI Agent Instruction Matrix**

| Parameter | Source Tag(s) & Path | Clinical Utility | Research Utility | Extraction Notes & Harmonization Strategy |
| :---- | :---- | :---- | :---- | :---- |
| **Treatment Modality** | (300A,00B0) \> (300A,00C4) BeamType, (300A,00B0) \> (300A,0111) ControlPointSequence | Distinguishes between 3D Conformal (STATIC), Intensity-Modulated Radiation Therapy (IMRT, DYNAMIC), and Volumetric Modulated Arc Therapy (VMAT, DYNAMIC with gantry motion). This informs clinicians about the complexity and conformality of the treatment. | A critical categorical feature for outcome models. Allows for stratification of patient cohorts by treatment technology. The modality is a known confounder in many analyses and must be accounted for. | **Instruction:** Check BeamType. If 'STATIC', classify as '3DCRT'. If 'DYNAMIC', further inspect the GantryAngle (300A,011E) within the ControlPointSequence. If the gantry angle changes between control points, classify as 'VMAT'. Otherwise, classify as 'IMRT'. Harmonize vendor-specific private tags (e.g., Varian's 'ARC' mode) to this classification. |
| **MLC Aperture Area & Complexity** | (300A,00B0) \> (300A,0111) \> (300A,011A) \> (300A,011C) LeafJawPositions | The shape of the beam, defined by the MLCs, conforms the high-dose region to the target and avoids OARs. The complexity of these shapes is a key determinant of plan quality and deliverability. | **Instruction:** For each control point, calculate the area of each leaf pair opening (distance between leaf positions multiplied by leaf width) and sum them to get the total aperture area. The variability of this area and the speed of leaf travel across control points can be used to compute plan complexity metrics like the Modulation Complexity Score (MCS). These metrics may correlate with treatment delivery time, accuracy, and clinical outcomes. | The LeafJawPositions tag contains a flattened list of paired floats. The pipeline must correctly parse these pairs. Leaf widths are not in the plan and must be known from the machine model specified in TreatmentMachineName (300A,00B2). |
| **Prescribed Dose per Fraction** | (300A,0070) \> (300A,0080) \> (300A,0084) TargetPrescriptionDose | This is the fundamental prescription: the dose to be delivered in each treatment session. Common values are 1.8, 2.0, or 2.2 Gy for conventional fractionation, or much higher for stereotactic treatments. | The primary independent variable for dose-response modeling. **Instruction:** Extract this value and the NumberOfFractionsPlanned (300A,0078) to calculate the total prescribed dose. Use these to compute standardized metrics like Biological Effective Dose (BED) and Equivalent Dose in 2-Gy fractions (EQD2) to compare different fractionation schemes. | The value is in Gray (Gy). The DoseReferenceDescription (300A,0026) often contains free text like "2 Gy/fx" which can be parsed for validation. Handle plans with multiple FractionGroupSequence items (e.g., boosts) by extracting the prescription for each. |
| **Delivered vs. Planned Meterset** | RTRECORD: (3008,0021) \> (3008,0042) DeliveredMeterset vs. RTPLAN: (300A,00B0) \> (300A,0086) BeamMeterset | A primary daily quality assurance check. A deviation greater than a set tolerance (e.g., 3-5%) indicates a potential machine fault or delivery error that requires immediate clinical investigation to ensure patient safety. | Quantifies real-world delivery accuracy. **Instruction:** For each delivered beam in the RTRECORD, link it to the planned beam in the RTPLAN via ReferencedBeamNumber. Calculate the percentage difference: . This deviation can be used as a feature in outcome models, representing a perturbation from the ideal plan. | The linkage requires parsing the ReferencedRTPlanSequence in the RTRECORD. The pipeline must have access to both the RTPLAN and RTRECORD files to perform this comparison. |
| **Target Volume (GTV, CTV, PTV)** | RTSTRUCT: (3006,0020) \> (3006,0026) ROIName, (3006,0039) ROIContourSequence | Defines the Gross Tumor Volume (GTV), Clinical Target Volume (CTV), and Planning Target Volume (PTV). The size, shape, and location of these volumes are fundamental to the entire treatment plan. | The volume of the target is a powerful prognostic factor. **Instruction:** Reconstruct the 3D mask for the target ROI. Calculate its volume in cubic centimeters (cm³). Use the mask for radiomics feature extraction. The shape and location can be used to derive geometric features (e.g., distance to nearest OAR). | **Harmonization:** ROIName is highly variable. The pipeline must implement a mapping dictionary or a rule-based system (e.g., using regular expressions) to map variants like "PTV\_50.4", "PTV5040", "Planning Target Volume" to a single standardized concept, 'PTV'. |

### **Enabling Advanced Research: Radiomics and Big Data**

A well-architected DICOM-RT pipeline is the gateway to advanced quantitative research, most notably radiomics. Radiomics is the process of extracting a large number of quantitative features from medical images, with the hypothesis that these features can reveal underlying tumor phenotypes and predict clinical outcomes.47

The data extracted by the pipeline directly enables the standard radiomics workflow:

1. **Image Ingestion:** The pipeline identifies and loads the primary DICOM image series (CT, MR, PET) referenced by the RT objects.  
2. **ROI Segmentation:** The pipeline uses the contour data from the RTSTRUCT file for a specific ROI (e.g., the GTV) to generate a 3D binary mask.49 This automated segmentation is far more scalable and reproducible than manual re-contouring for research.  
3. **Feature Extraction:** Using a library like pyradiomics, the pipeline can then calculate hundreds of quantitative features from the image voxels that lie within the 3D mask.50 These features describe the tumor's intensity, shape, and texture.  
4. **Data Correlation:** The extracted features can then be correlated with clinical outcomes (e.g., survival, recurrence) and other data extracted from the DICOM files, such as the prescribed dose from the RTPLAN or plan complexity metrics derived from the MLC positions. A particularly advanced application is "dose-omics," where the 3D dose grid from the RTDOSE is overlaid with the image and ROI mask to analyze how dose patterns within the tumor relate to radiomic feature changes and patient outcomes. This entire process is critically dependent on the spatial integrity guaranteed by a shared Frame of Reference UID across all involved objects.

The structured, harmonized data produced by the pipeline is also essential for building large-scale "big data" repositories for radiation oncology. By extracting and normalizing key dosimetric, geometric, and clinical parameters from thousands of patient records, researchers can build powerful predictive models for treatment toxicity, tumor control, and overall survival. Initiatives and policies from organizations like ASTRO encourage the sharing of such de-identified data to accelerate research in the field.51

## **Conclusion: Building a Future-Proof Radiotherapy Data Asset**

The development of a comprehensive data extraction pipeline for DICOM-RT files is a complex but highly valuable undertaking. Success requires moving beyond a simple tag-dumping approach and embracing a more sophisticated, context-aware strategy. This report has outlined a blueprint for such a strategy, emphasizing several core principles.

First, the pipeline's architecture should mirror the clinical workflow, treating the DICOM-RT objects as an interconnected data narrative. The use of UIDs, particularly the Frame of Reference UID, must be central to the design to ensure the logical and spatial integrity of the extracted data.

Second, the pipeline must be built with the awareness that vendor implementations of the DICOM standard are not uniform. It requires a deep dive into vendor-specific DICOM Conformance Statements to understand how standard tags are used and to identify valuable private tags. For some ecosystems, like Varian's, a hybrid approach that combines DICOM parsing with vendor-specific APIs like ESAPI may be necessary for truly comprehensive data acquisition.

Third, the pipeline's function is not merely to extract but to harmonize. It must contain a logic layer that translates the heterogeneous, often free-text, data found in clinical DICOM files into a standardized, research-grade ontology. This harmonization is the critical step that transforms a collection of data into a powerful asset for large-scale analytics, machine learning, and radiomics.

Finally, the pipeline must be designed for the future. The DICOM standard is not static; the development of second-generation RT objects aims to address many of the complexities of the current standard, such as the monolithic RTPLAN object and the challenges of representing adaptive radiotherapy.19 A well-designed pipeline, with a clear separation between raw data extraction, harmonization logic, and the final data model, will be more adaptable to these future standards. By adhering to these principles, the developer can create not just a data extraction tool, but a future-proof, scalable, and research-ready radiotherapy data platform capable of fueling the next generation of clinical and scientific discovery.

#### **Works cited**

1. dicom in radiotherapy \- Swiss Society of Radiobiology and Medical Physics, accessed October 4, 2025, [https://ssrpm.ch/dicom-in-radiotherapy/](https://ssrpm.ch/dicom-in-radiotherapy/)  
2. DICOM IN RADIOTHERAPY \- NEMA, accessed October 4, 2025, [https://dicom.nema.org/dicom/geninfo/brochure/rtaapm.htm](https://dicom.nema.org/dicom/geninfo/brochure/rtaapm.htm)  
3. DICOM-RT and Its Utilization in Radiation Therapy | RadioGraphics \- RSNA Journals, accessed October 4, 2025, [https://pubs.rsna.org/doi/abs/10.1148/rg.293075172](https://pubs.rsna.org/doi/abs/10.1148/rg.293075172)  
4. RT\_Beams Treatment Record \- Society for Imaging Informatics in Medicine, accessed October 4, 2025, [https://siim.org/otpedia/rt\_beams-treatment-record/](https://siim.org/otpedia/rt_beams-treatment-record/)  
5. Add and Modify ROIs of DICOM-RT Contour Data \- MATLAB & Simulink \- MathWorks, accessed October 4, 2025, [https://www.mathworks.com/help/images/add-and-modify-rois-of-dicomrt-contour-data.html](https://www.mathworks.com/help/images/add-and-modify-rois-of-dicomrt-contour-data.html)  
6. DICOM Radiotherapy Structure Sets | IDC User Guide, accessed October 4, 2025, [https://learn.canceridc.dev/dicom/derived-objects/dicom-radiotherapy-structure-sets](https://learn.canceridc.dev/dicom/derived-objects/dicom-radiotherapy-structure-sets)  
7. C.8.8 Radiotherapy Modules \- DICOM, accessed October 4, 2025, [https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect\_c.8.8.html](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_c.8.8.html)  
8. Digital Imaging and Communications in Medicine (DICOM) Supplement 11 Radiotherapy Objects, accessed October 4, 2025, [https://www.dicomstandard.org/News-dir/ftsup/docs/sups/sup11.pdf](https://www.dicomstandard.org/News-dir/ftsup/docs/sups/sup11.pdf)  
9. RT Plan \- Society for Imaging Informatics in Medicine, accessed October 4, 2025, [https://siim.org/otpedia/rt-plan/](https://siim.org/otpedia/rt-plan/)  
10. RT Dose \- Society for Imaging Informatics in Medicine, accessed October 4, 2025, [https://siim.org/otpedia/rt-dose/](https://siim.org/otpedia/rt-dose/)  
11. C.8.8.3 RT Dose Module \- DICOM, accessed October 4, 2025, [https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect\_c.8.8.3.html](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_c.8.8.3.html)  
12. DICOM RT BeamsTreatmentRecord \- DCMTK \- OFFIS DCMTK and ..., accessed October 4, 2025, [https://support.dcmtk.org/redmine/projects/dcmtk/wiki/DICOM\_RT\_BeamsTreatmentRecord](https://support.dcmtk.org/redmine/projects/dcmtk/wiki/DICOM_RT_BeamsTreatmentRecord)  
13. DICOM: Definitions and Testing \- AAPM, accessed October 4, 2025, [https://www.aapm.org/meetings/04AM/pdf/14-2274-47948.pdf](https://www.aapm.org/meetings/04AM/pdf/14-2274-47948.pdf)  
14. RT Plan Geometry Attribute – DICOM Standard Browser, accessed October 4, 2025, [https://dicom.innolitics.com/ciods/rt-plan/rt-general-plan/300a000c](https://dicom.innolitics.com/ciods/rt-plan/rt-general-plan/300a000c)  
15. RT Dose Module – DICOM Standard Browser, accessed October 4, 2025, [https://dicom.innolitics.com/ciods/rt-dose/rt-dose](https://dicom.innolitics.com/ciods/rt-dose/rt-dose)  
16. Three different RTDOSE in the same RTPLAN \- Support \- 3D Slicer Community, accessed October 4, 2025, [https://discourse.slicer.org/t/three-different-rtdose-in-the-same-rtplan/25576](https://discourse.slicer.org/t/three-different-rtdose-in-the-same-rtplan/25576)  
17. Generating RT Structs in Python with RT-UTILS \- BC Cancer Research, accessed October 4, 2025, [https://www.bccrc.ca/dept/io-programs/qurit/blog/generating-rt-structs-python-rt-utils](https://www.bccrc.ca/dept/io-programs/qurit/blog/generating-rt-structs-python-rt-utils)  
18. RT-Utils: A minimal Python library for RT-Struct manipulation \- BC Cancer Research, accessed October 4, 2025, [https://www.bccrc.ca/dept/io-programs/qurit/software/rt-utils-minimal-python-library-rt-struct-manipulation](https://www.bccrc.ca/dept/io-programs/qurit/software/rt-utils-minimal-python-library-rt-struct-manipulation)  
19. Next Generation of the DICOM Standard for Radiation Therapy, accessed October 4, 2025, [https://www.dicomstandard.org/docs/librariesprovider2/dicomdocuments/wp-cotent/uploads/2018/10/day2\_s14-busch-dicom-rt-2nd-generation-v2.pdf?sfvrsn=25f503d6\_2](https://www.dicomstandard.org/docs/librariesprovider2/dicomdocuments/wp-cotent/uploads/2018/10/day2_s14-busch-dicom-rt-2nd-generation-v2.pdf?sfvrsn=25f503d6_2)  
20. A.29 RT Beams Treatment Record IOD \- DICOM, accessed October 4, 2025, [https://dicom.nema.org/dicom/2013/output/chtml/part03/sect\_A.29.html](https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.29.html)  
21. Digital Imaging and Communications in Medicine (DICOM) Supplement 29: Radiotherapy Treatment Records and Radiotherapy Media Exte, accessed October 4, 2025, [https://www.dicomstandard.org/News-dir/ftsup/docs/sups/sup29.pdf](https://www.dicomstandard.org/News-dir/ftsup/docs/sups/sup29.pdf)  
22. DICOM Statements | Varian, accessed October 4, 2025, [https://www.varian.com/why-varian/interoperability/dicom-statements](https://www.varian.com/why-varian/interoperability/dicom-statements)  
23. Dicom formats supported by Eclipse : r/MedicalPhysics \- Reddit, accessed October 4, 2025, [https://www.reddit.com/r/MedicalPhysics/comments/1esjyyb/dicom\_formats\_supported\_by\_eclipse/](https://www.reddit.com/r/MedicalPhysics/comments/1esjyyb/dicom_formats_supported_by_eclipse/)  
24. DICOM Conformance Statement \- Elekta, accessed October 4, 2025, [https://www.elekta.com/products/oncology-informatics/dicom-conformance-statements/documents/Focal-Monaco-Release-Focal-480-Monaco-330-Document-LEDDCMFCLMON000170.pdf](https://www.elekta.com/products/oncology-informatics/dicom-conformance-statements/documents/Focal-Monaco-Release-Focal-480-Monaco-330-Document-LEDDCMFCLMON000170.pdf)  
25. DICOM Conformance Statement, accessed October 4, 2025, [https://www.nbwhprint.com/de/dam/jcr:fdfb8786-e78d-4b40-b21f-debfc0b14a8c/DICOM-Conformance-Statement-LGP-11-0.pdf](https://www.nbwhprint.com/de/dam/jcr:fdfb8786-e78d-4b40-b21f-debfc0b14a8c/DICOM-Conformance-Statement-LGP-11-0.pdf)  
26. DICOM Conformance Statement \- Elekta, accessed October 4, 2025, [https://www.elekta.com/products/oncology-informatics/dicom-conformance-statements/documents/DICOM-Conformance-Statement-LGP-11-3.pdf](https://www.elekta.com/products/oncology-informatics/dicom-conformance-statements/documents/DICOM-Conformance-Statement-LGP-11-3.pdf)  
27. DICOM Conformance Statements | Oncology Informatics | Elekta, accessed October 4, 2025, [https://www.elekta.com/products/oncology-informatics/dicom-conformance-statements/](https://www.elekta.com/products/oncology-informatics/dicom-conformance-statements/)  
28. Varian System Server v18 | PDF | Computing \- Scribd, accessed October 4, 2025, [https://www.scribd.com/document/785106656/Varian-System-Server-v18](https://www.scribd.com/document/785106656/Varian-System-Server-v18)  
29. DICOM Conformance Statement \- Elekta, accessed October 4, 2025, [https://www.elekta.com/products/oncology-informatics/dicom-conformance-statements/documents/DICOM-Conformance-Statement-LGP-11-4.pdf](https://www.elekta.com/products/oncology-informatics/dicom-conformance-statements/documents/DICOM-Conformance-Statement-LGP-11-4.pdf)  
30. MOSAIQ 2.60 DICOM Conformance Statement | Elekta, accessed October 4, 2025, [https://www.elekta.com/products/oncology-informatics/dicom-conformance-statements/documents/LEDDCM0048.pdf](https://www.elekta.com/products/oncology-informatics/dicom-conformance-statements/documents/LEDDCM0048.pdf)  
31. ARIA Oncology Information System \- Varian, accessed October 4, 2025, [https://www.varian.com/products/software/digital-oncology/oncology-management-systems/aria-oncology-information-system](https://www.varian.com/products/software/digital-oncology/oncology-management-systems/aria-oncology-information-system)  
32. Dicom Confrormance | PDF | Interoperability | Information Retrieval \- Scribd, accessed October 4, 2025, [https://www.scribd.com/document/59174493/dicom-confrormance](https://www.scribd.com/document/59174493/dicom-confrormance)  
33. Documenation Template \- Accuray, accessed October 4, 2025, [https://www.accuray.com/wp-content/uploads/10640071.pdf](https://www.accuray.com/wp-content/uploads/10640071.pdf)  
34. Documenation Template \- Accuray, accessed October 4, 2025, [https://www.accuray.com/wp-content/uploads/1060655.pdf](https://www.accuray.com/wp-content/uploads/1060655.pdf)  
35. VarianAPIs/PyESAPI: Python interface to Eclipse Scripting API \- GitHub, accessed October 4, 2025, [https://github.com/VarianAPIs/PyESAPI](https://github.com/VarianAPIs/PyESAPI)  
36. About the Eclipse Scripting API | Varian Innovation Center Documentation Hub, accessed October 4, 2025, [https://docs.developer.varian.com/articles/16.1/04\_About\_the\_Eclipse\_Scripting\_API.html](https://docs.developer.varian.com/articles/16.1/04_About_the_Eclipse_Scripting_API.html)  
37. MOSAIQ Real-World Testing | Oncology Informatics | Software \- Elekta, accessed October 4, 2025, [https://www.elekta.com/products/oncology-informatics/mosaiq-real-world-testing/](https://www.elekta.com/products/oncology-informatics/mosaiq-real-world-testing/)  
38. Medical Oncology | Oncology Informatics | Software | Elekta, accessed October 4, 2025, [https://www.elekta.com/products/oncology-informatics/elekta-one/oncology-care/medical-oncology/](https://www.elekta.com/products/oncology-informatics/elekta-one/oncology-care/medical-oncology/)  
39. Elekta Solutions AB February 23, 2023 Melinda Smith Director, Regulatory Affairs & Quality \- accessdata.fda.gov, accessed October 4, 2025, [https://www.accessdata.fda.gov/cdrh\_docs/pdf22/K223229.pdf](https://www.accessdata.fda.gov/cdrh_docs/pdf22/K223229.pdf)  
40. MOSAIQ and Monaco Interoperability \- Atlas Medical LLC, accessed October 4, 2025, [https://atlasmedical.ai/wp-content/uploads/2023/09/MOSAIQ-and-Monaco-Interoperability-Leaflet-3.pdf](https://atlasmedical.ai/wp-content/uploads/2023/09/MOSAIQ-and-Monaco-Interoperability-Leaflet-3.pdf)  
41. Private Data Elements — pydicom 3.1.0.dev0 documentation, accessed October 4, 2025, [http://pydicom.github.io/pydicom/dev/guides/user/private\_data\_elements.html](http://pydicom.github.io/pydicom/dev/guides/user/private_data_elements.html)  
42. Is there any way to get private tags in pydicom? \- Stack Overflow, accessed October 4, 2025, [https://stackoverflow.com/questions/45590603/is-there-any-way-to-get-private-tags-in-pydicom](https://stackoverflow.com/questions/45590603/is-there-any-way-to-get-private-tags-in-pydicom)  
43. API Documentation \- Oncology Informatics \- Elekta, accessed October 4, 2025, [https://www.elekta.com/products/oncology-informatics/interoperability/fhir-innovation/api/](https://www.elekta.com/products/oncology-informatics/interoperability/fhir-innovation/api/)  
44. SU-E-T-33: pydicom: an open source DICOM library \- ResearchGate, accessed October 4, 2025, [https://www.researchgate.net/publication/258538185\_SU-E-T-33\_pydicom\_an\_open\_source\_DICOM\_library](https://www.researchgate.net/publication/258538185_SU-E-T-33_pydicom_an_open_source_DICOM_library)  
45. Private Data Elements — pydicom 3.0.1 documentation, accessed October 4, 2025, [https://pydicom.github.io/pydicom/stable/guides/user/private\_data\_elements.html](https://pydicom.github.io/pydicom/stable/guides/user/private_data_elements.html)  
46. TG-263: Standardizing Nomenclatures in Radiation Oncology \- AAPM, accessed October 4, 2025, [https://www.aapm.org/pubs/reports/RPT\_263.pdf](https://www.aapm.org/pubs/reports/RPT_263.pdf)  
47. Machine and Deep Learning Methods for Radiomics \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8965689/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8965689/)  
48. Computational resources for radiomics \- Court \- Translational Cancer Research, accessed October 4, 2025, [https://tcr.amegroups.org/article/view/8409/html](https://tcr.amegroups.org/article/view/8409/html)  
49. Prepare data and ROI for radiomics feature extraction \- MATLAB \- MathWorks, accessed October 4, 2025, [https://www.mathworks.com/help/medical-imaging/ref/radiomics.html](https://www.mathworks.com/help/medical-imaging/ref/radiomics.html)  
50. radiomics-feature-extraction · GitHub Topics, accessed October 4, 2025, [https://github.com/topics/radiomics-feature-extraction](https://github.com/topics/radiomics-feature-extraction)  
51. ASTRO Journals' Data Sharing Policy and Recommended Best Practices \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6817515/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6817515/)  
52. ASTRO Journals' Data Sharing Policy and Recommended Best Practices \- ResearchGate, accessed October 4, 2025, [https://www.researchgate.net/publication/335325550\_ASTRO\_Journals'\_Data\_Sharing\_Policy\_and\_Recommended\_Best\_Practices](https://www.researchgate.net/publication/335325550_ASTRO_Journals'_Data_Sharing_Policy_and_Recommended_Best_Practices)