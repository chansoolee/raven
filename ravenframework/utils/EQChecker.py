# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, re
import numpy as np
from xml.etree import ElementTree as ET
from ravenframework.utils import randomUtils

class EQChecker():
  """
  EQ checker and genome modifier for existing EQ cycle genomes.
  """
  def __init__(self,prloDataInputFile):
    """
      Initializing variables.
      @ In, None
      @ Out, None
    """
    self.prloData = self.PRLODataParser(prloDataInputFile)

  ## function for shuffling scheme logic
  def checkGenome(self,genome,symMult):
    """
      @ In, genome, list, the faID selections that define the current solution.
      @ In, zoneMap, list, zoneMap with the batch assignements at each location in the symmetric solution.
      @ In, symMult, dict, multiplicity of each location in the zoning map (1-indexed).
      @ Out, bool, False if any check is violated; True otherwise.
    """
    decodedGenomeWithMult = [(self.decodeFAID(genome[l],self.prloData.solnLen,self.prloData.numBatches),symMult[l+1]) for l in range(len(genome))]
    zoneMap = [l[0][1] for l in decodedGenomeWithMult]

  ## Assert: no batch is larger than the previous one.
    batchCount = {}
    for batNum in set(zoneMap):
      batchCount[batNum] = sum([symMult[i+1] for i in range(len(zoneMap)) if zoneMap[i] == batNum])
    for batNum in batchCount.keys():
      if batNum == 1:
        continue
      else:
        if batchCount[batNum] > batchCount[batNum-1]:
          return False

  ## Assert: every FA has a valid source
    for gene in genome:
      currentFA = self.decodeFAID(gene, self.prloData.solnLen, self.prloData.numBatches)
      if currentFA[1] == 1: #fresh fuel FA
        continue
      else:
        sourceFA = self.decodeFAID(genome[currentFA[0]-1], self.prloData.solnLen, self.prloData.numBatches)
        # check that FA types match and current batchNum = source batchNum + 1
        if currentFA[2] != sourceFA[2] or currentFA[1] - sourceFA[1] != 1:
          return False

  ## Assert: no shuffling motion violates multiplicity
    for i in range(len(decodedGenomeWithMult)):
      gene = decodedGenomeWithMult[i]
      reloadedCount = sum([loc[1] for loc in decodedGenomeWithMult if loc[0] == (i+1, gene[0][1]+1, gene[0][2])])
      if reloadedCount > gene[1]:
        return False

  ## Assert: FA type doesn't change during reshuffle and batch number increments correctly.
    for i in range(len(decodedGenomeWithMult)):
      gene = decodedGenomeWithMult[i]
      if gene[0][2] != decodedGenomeWithMult[gene[0][0]-1][0][2]: # 1-indexed to 0-indexed
        return False
      elif decodedGenomeWithMult[gene[0][0]-1][0][1] - gene[0][1] == 1:
        return False

    return True

  class PRLODataParser():
    """
      Interprets the data from the PRLO data file in '.xml'.
      This is a trimmed down version of the PARCSv345InpGen.PRLODataParser class.
      #!NOTE(rollnk): it is likely more elegant to have a single, external version of this class rather than redefine it repeatedly.
    """
    def __init__(self, inputFile, verbosity="full"):
      """
        Constructor.
        @ In, inputFile, string, xml PARCSv345 parameters file
        @ Out, None
      """
      fullFile = os.path.join(inputFile)
      dorm = ET.parse(fullFile)
      root = dorm.getroot()

      # If elements are nested under a <PARCS> section, use that as the search root.
      parcsNode = root.find('PARCS')
      if parcsNode is not None:
        root = parcsNode

      # Parse user-provided data from XML file
      #!TODO: define default values for missing params; list expected formats/units here.
      if verbosity in ['calcType','full','reduced']:
        self.calculationType = '_'.join(root.find('calculationType').text.strip().lower().split()) if root.find('calculationType') is not None else "single_cycle"
      if verbosity in ['full','reduced']:
        self.numAssemblies = int(root.find('numAssemblies').text.strip())
        self.numBatches = int(root.find('numBatches').text.strip())
        self.feedBatchSizeLimits = root.find('feedBatchSizeLimits').text.strip() if root.find('feedBatchSizeLimits') is not None else None
        self.colLabels = root.find('colLabels').text.strip()
        self.rowLabels = root.find('rowLabels').text.strip()
        self.geometry = root.find('geometry').text.strip()
        self.coreShape = re.sub(r"r\d", '0', re.sub(r"\d+", '1', self.geometry.replace('00', ' ')))
        self.solnLen = max([int(s) for s in self.geometry.split() if s.isdigit()])
        self.faDict = []
        for fa in root.iter('FA'):
          self.faDict.append(fa.attrib)
        self.numTypes = len([fa for fa in self.faDict if fa['type'] == 'fuel'])
        self.xsDict =[]
        for xs in root.iter('XS'):
          self.xsDict.append(xs.attrib)

        # Resolve calculated values from user-provided data
        if not self.feedBatchSizeLimits:
          self.feedBatchSizeLimits = (np.ceil(self.numAssemblies/self.numBatches),self.numAssemblies) # logical extremes for feed batch size
        else:
          self.feedBatchSizeLimits = tuple([int(val.strip()) for val in self.feedBatchSizeLimits.replace(',',' ').split()])
          if len(self.feedBatchSizeLimits) == 1:
            # if only one value is given, assume that value is the maximum limit.
            self.feedBatchSizeLimits = (np.ceil(self.numAssemblies/self.numBatches),self.feedBatchSizeLimits[0])
          else:
            self.feedBatchSizeLimits = (self.feedBatchSizeLimits[0],self.feedBatchSizeLimits[-1]) #ensure only two values are provided.

      if verbosity in ['full']:
        self.useTemplate = self.str_to_bool(root.find('useTemplate').text.strip()) if root.find('useTemplate') is not None else False
        self.THFlag = self.str_to_bool(root.find('THFlag').text.strip()) if root.find('THFlag') is not None else False
        self.power = float(root.find('power').text.strip()) if root.find('power') is not None else 100.0
        self.coreType = root.find('coreType').text.strip().upper() if root.find('coreType') is not None else "PWR"
        self.initialExposure = float(root.find('initialExposure').text.strip()) if root.find('initialExposure') is not None else 0.00
        self.initialBoron = int(root.find('initialBoron').text.strip())
        self.pinPowerRecFlag = self.str_to_bool(root.find('pinPowerRecFlag').text.strip())
        self.numAxial = int(root.find('numAxial').text.strip())
        self.BC = root.find('BC').text.strip()
        self.faPower = float(root.find('faPower').text.strip())
        self.faPitch = float(root.find('faPitch').text.strip())
        self.inletTemp = float(root.find('inletTemp').text.strip())
        self.flow = float(root.find('flow').text.strip()) #!TODO: I believe this is mass flow; doublecheck
        self.depHistory = root.find('depHistory').text.strip()
        self.inpHistFile = root.find('inpHistFile').text.strip() if root.find('inpHistFile') is not None else None
        self.xsLib = root.find('xsLib').text.strip()
        self.xsExtension = root.find('xsExtension').text.strip()

        fuelMap = [int(s) for s in self.geometry.split() if s.isdigit()]
        self.symmetricMultiplicity = {i:fuelMap.count(i) for i in fuelMap}
        del self.symmetricMultiplicity[0] #locations are 1-indexed; '0' is the void space.

    def str_to_bool(self,string):
      if string.lower() in ['t','true','1','yes','y']:
        return True
      elif string.lower() in ['f','false','0','no','n']:
        return False
      else:
        raise ValueError(f"Failed to convert string '{string}' to boolean.")

  def decodeFAID(self,faID,solnLen,numBatches):
    """
      PRLO uses a numbering scheme for the solution encoding that assumes
      all possible FA's are by batch, then location, then type, for a total
      count of possible FA designations = numBatches*numAssemblies*numTypes
      @ In, faID, str, the FA ID according to the PRLO encoding scheme (0-indexed)
      @ In, solnLen, int, the number of fuel locations in the solution
      @ In, numBatches, int, the maximum number of fuel batches.
      @ Out, locNum, int, source location in the solution for faID (1-indexed)
      @ Out, batchNum, int, batch number for faID (1-indexed)
      @ Out, typeNum, int, FA type number for faID (1-indexed)
    """
    # decode location
    locNum = int(np.ceil(((faID + 1) % (solnLen*numBatches)) / numBatches))
    if locNum == 0:
      locNum = int(np.ceil((faID % (solnLen*numBatches)) / numBatches))
    # decode batch number
    batchNum = 1 + int(faID % numBatches)
    # decode FA type
    typeNum = 1 + int(faID / (solnLen * numBatches))
    return (locNum, batchNum, typeNum)

  def encodeFAID(self,FA,solnLen,numBatches):
    """
      PRLO uses a numbering scheme for the solution encoding that assumes
      all possible FA's are by batch, then location, then type, for a total
      count of possible FA designations = numBatches*numAssemblies*numTypes
      @ In, FA, tuple, the FA source location, batch number, and type (all 1-indexed)
      @ In, solnLen, int, the number of fuel locations in the solution
      @ In, numBatches, int, the maximum number of fuel batches.
      @ Out, faID, str, the FA ID according to the PRLO encoding scheme (0-indexed)
    """
    if FA[0] > solnLen or FA[1] > numBatches:
      return None # prevent eroneous FA's from being given ID's.
    else:
      return (FA[0]-1)*numBatches + solnLen*numBatches*(FA[2]-1) + (FA[1]-1)