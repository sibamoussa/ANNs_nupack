'''import statements'''
import numpy as np
import scipy.io
import random
from utils import *
import sys
import tqdm
from trainnupack import * 
try: # we don't always install these on every platform
    from nupack import *
except:
    pass

'''
This script computes a binding score for a given sequence or set of sequences

> Inputs: numpy integer arrays - different oracles with different requirements
> Outputs: oracle outputs - usually numbers

config
'dataset seed' - self explanatory
'dict size' - number of possible states per sequence element - e.g., for ATGC 'dict size' = 4
'variable sample length', 'min sample length', 'max sample length' - for determining the length and variability of sample sequences
'init dataset length' - number of samples for initial (random) dataset
'dataset' - name of dataset to be saved
'''


class Oracle():
    def __init__(self, config):
        '''
        initialize the oracle
        :param config:
        '''
        self.config = config
        self.seqLen = 30 # self.config.max_sample_length

        self.initRands()


    def initRands(self):
        '''
        initialize random numbers for custom-made toy functions
        :return:
        '''
        np.random.seed(self.config.toy_oracle_seed)

        # set these to be always positive to play nice with gFlowNet sampling
        if True:#self.config.test_mode:
            self.linFactors = -np.ones(self.seqLen) # Uber-simple function, for testing purposes - actually nearly functionally identical to one-max, I believe
        else:
            self.linFactors = np.abs(np.random.randn(self.seqLen))  # coefficients for linear toy energy

        hamiltonian = np.random.randn(self.seqLen,self.seqLen) # energy function
        self.hamiltonian = np.tril(hamiltonian) + np.tril(hamiltonian, -1).T # random symmetric matrix

        pham = np.zeros((self.seqLen,self.seqLen,self.config.dataset_dict_size,self.config.dataset_dict_size))
        for i in range(pham.shape[0]):
            for j in range(i, pham.shape[1]):
                for k in range(pham.shape[2]):
                    for l in range(k, pham.shape[3]):
                        num =  - np.random.uniform(0,1)
                        pham[i, j, k, l] = num
                        pham[i, j, l, k] = num
                        pham[j, i, k, l] = num
                        pham[j, i, l, k] = num
        self.pottsJ = pham # multilevel spin Hamiltonian (Potts Hamiltonian) - coupling term
        self.pottsH = np.random.randn(self.seqLen,self.config.dataset_dict_size) # Potts Hamiltonian - onsite term

        # W-model parameters
        # first get the binary dimension size
        aa = np.arange(self.config.dataset_dict_size)
      
        x0 = np.binary_repr(aa[-1])
        dimension = int(len(x0) * self.config.dataset_size)

        mu = np.random.randint(1, dimension + 1)
        v = np.random.randint(1, dimension + 1)
        m = np.random.randint(1, dimension)
        n = np.random.randint(1, dimension)
        gamma = np.random.randint(0, int(n * (n - 1 ) / 2))
        self.mu, self.v, self.m, self.n, self.gamma = [mu, v, m, n, gamma]


    def initializeDataset(self,save = True, returnData = False, customSize=None):
        '''
        generate an initial toy dataset with a given number of samples
        need an extra factor to speed it up (duplicate filtering is very slow)
        :param numSamples:
        :return:
        '''
        data = {}
        np.random.seed(self.config.dataset_seed)
        if customSize is None:
            datasetLength = self.config.dataset_size
        else:
            datasetLength = customSize

      
        samples = np.random.randint(1, self.config.dataset_dict_size + 1,size=(datasetLength, self.config.max_sample_length))
        samples = filterDuplicateSamples(samples)
        while len(samples) < datasetLength:
                samples = np.concatenate((samples,np.random.randint(1, self.config.dataset_dict_size + 1, size=(datasetLength, self.config.max_sample_length))),0)
                samples = filterDuplicateSamples(samples)

        data['samples'] = samples
        data['scores'] = self.score(data['samples'])
        
        if save:
            np.save('nupack_dataset', data)
        if returnData:
            return data
    def score(self, queries):
        '''
        assign correct scores to selected sequences
        :param queries: sequences to be scored
        :return: computed scores
        '''
        if isinstance(queries, list):
            queries = np.asarray(queries) # convert queries to array
        block_size = int(1e4) # score in blocks of maximum 10000
        scores_list = []
        scores_dict = {}
        for idx in tqdm.tqdm(range(len(queries) // block_size + bool(len(queries) % block_size))):
            queryBlock = queries[idx * block_size:(idx + 1) * block_size]
            scores_block = self.getScore(queryBlock)
            if isinstance(scores_block, dict):
                for k, v in scores_block.items():
                    if k in scores_dict:
                        scores_dict[k].extend(list(v))
                    else:
                        scores_dict.update({k: list(v)})
            else:
                scores_list.extend(self.getScore(queryBlock))
        if len(scores_list) > 0:
            return np.asarray(scores_list)
        else:
            return {k: np.asarray(v) for k, v in scores_dict.items()}


    def getScore(self,queries):
        if self.config.oracle == 'nupack energy':
            return self.nupackScore(queries, returnFunc = 'energy')
        elif self.config.oracle == 'nupack pins':
            return -self.nupackScore(queries, returnFunc = 'pins')
        elif self.config.oracle == 'nupack pairs':
            return -self.nupackScore(queries, returnFunc = 'pairs')
        elif isinstance(self.config.oracle, list) and all(["nupack " in el for el in self.config.dataset_oracle]):
            return self.nupackScore(queries, returnFunc=[el.replace("nupack ", "") for el in self.config.oracle])
        else:
            raise NotImplementedError("Unknown orackle type")

    def numbers2letters(self, sequences):  # Tranforming letters to numbers (1234 --> ATGC)
        '''
        Converts numerical values to ATCG-format
        :param sequences: numerical DNA sequences to be converted
        :return: DNA sequences in ATCG format
        '''
        if type(sequences) != np.ndarray:
            sequences = np.asarray(sequences)

        my_seq = ["" for x in range(len(sequences))]
        row = 0
        for j in range(len(sequences)):
            seq = sequences[j, :]
            assert type(seq) != str, 'Function inputs must be a list of equal length strings'
            for i in range(len(sequences[0])):
                na = seq[i]
                if na == 1:
                    my_seq[row] += 'A'
                elif na == 2:
                    my_seq[row] += 'T'
                elif na == 3:
                    my_seq[row] += 'C'
                elif na == 4:
                    my_seq[row] += 'G'
            row += 1
        return my_seq


    def numpy_fillna(self, data):
        '''
        function to pad uneven-length vectors up to the max with zeros
        :param data:
        :return:
        '''
        # Get lengths of each row of data
        lens = np.array([len(i) for i in data])

        # Mask of valid places in each row
        mask = np.arange(lens.max()) < lens[:, None]

        # Setup output array and put elements from data into masked positions
        out = np.zeros(mask.shape, dtype=object)
        out[mask] = np.concatenate(data)
        return out


    def nupackScore(self, queries, returnFunc='energy'):
        # Nupack requires Linux OS.
        #use nupack instead of seqfold - more stable and higher quality predictions in general
        #returns the energy of the most probable structure only
        #:param queries:
        #:param returnFunct 'energy' 'pins' 'pairs'
        #:return:

        temperature = 310.0  # Kelvin
        ionicStrength = 1.0 # molar
        sequences = self.numbers2letters(queries)

        energies = np.zeros(len(sequences))
        strings = []
        nPins = np.zeros(len(sequences)).astype(int)
        nPairs = 0
        ssStrings = np.zeros(len(sequences),dtype=object)

        # parallel evaluation - fast
        strandList = []
        comps = []
        i = -1
        for sequence in sequences:
            i += 1
            strandList.append(Strand(sequence, name='strand{}'.format(i)))
            comps.append(Complex([strandList[-1]], name='comp{}'.format(i)))

        set = ComplexSet(strands=strandList, complexes=SetSpec(max_size=1, include=comps))
        model1 = Model(material='dna', celsius=temperature - 273, sodium=ionicStrength)
        results = complex_analysis(set, model=model1, compute=['mfe'])
        for i in range(len(energies)):
            energies[i] = results[comps[i]].mfe[0].energy
            ssStrings[i] = str(results[comps[i]].mfe[0].structure)

        dict_return = {}
        if 'pins' in returnFunc:
            for i in range(len(ssStrings)):
                indA = 0  # hairpin completion index
                for j in range(len(sequences[i])):
                    if ssStrings[i][j] == '(':
                        indA += 1
                    elif ssStrings[i][j] == ')':
                        indA -= 1
                        if indA == 0:  # if we come to the end of a distinct hairpin
                            nPins[i] += 1
            dict_return.update({"pins": nPins})
        if 'pairs' in returnFunc:
            nPairs = np.asarray([ssString.count('(') for ssString in ssStrings]).astype(int)
            dict_return.update({"pairs": nPairs})
        if 'energy' in returnFunc:
            dict_return.update({"energy": energies})

        if isinstance(returnFunc, list):
            if len(returnFunc) > 1:
                return dict_return
            else:
                return dict_return[returnFunc[0]]
        else:
            return dict_return[returnFunc]
