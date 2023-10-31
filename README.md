
# PANDA-3D: protein function prediction based on AlphaFold models

This repository includes code and a pre-trained model of PANDA-3D for protein function prediction.

# Protein function prediction
PANDA-3D can predict protein function using AlphaFold-predicted structures. PANDA-3D is offered as a stand-alone tool and a web server.

## Stand-alone PANDA-3D
1. Download the trained model to [panda3code/](panda3code/)
```
wget http://dna.cs.miami.edu/PANDA-3D/download_files/trained.model panda3code/
```
2. Download a predicted structure to [example/](example/) from the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/). The protein "INTS1" with UniProtID "A0A6J7ZWZ8" has already been downloaded.
3. Run PANDA-3D
```
>> python run_PANDA-3D.py example/
```

>**Dependencies**  
>*PANDA-3D* is built under Python 3.10.  
>The conda environment is shared via "[conda_environment.yml](conda_environment.yml)".

## Web-server PANDA-3D
Submit predicted structures at
http://dna.cs.miami.edu/PANDA-3D/

## Output explaination:
The output is saved in [example/prediction.txt](example/prediction.txt). The file format is as follows:  
- The first two lines contain model information.  
- Subsequent lines follow this format:  
  - The first column is the name of the PDB structure file.  
  - The second column is the GO term ID.  
  - The third column is the confidence score predicted by PANDA-3D.  
- The last line contains only "END."
```txt
MODEL 1
KEYWORDS sequence embedding, geometric vector perceptron, transformer.
AF-A0A6J7ZWZ8-F1-model_v4       GO:0003006      0.93
AF-A0A6J7ZWZ8-F1-model_v4       GO:0010605      0.93
AF-A0A6J7ZWZ8-F1-model_v4       GO:0070727      0.92
...
AF-A0A6J7ZWZ8-F1-model_v4       GO:0016798      0.09
END
```
## Citation
```
@article{zhao2023PANDA3D,
  title={PANDA-3D: protein function prediction based on
AlphaFold models},
  author={Zhao, Chenguang and Liu, Tong and Wang, Zheng},
  publisher={Under Review}
}
```