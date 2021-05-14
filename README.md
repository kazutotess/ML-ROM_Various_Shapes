# Machine-learning-based reduced-order modeling for unsteady flows around bluff bodies of various shapes
This repository contains the simple source codes of "Machine-learning-based reduced-order modeling for unsteady flows around bluff bodies of various shapes," [Theor. Comput. Fluid Dyn. 34, 367-383 (2020).][thesis] (Preprint: [arXiv:2003.07548 [physics.flu-dyn]][airxiv])
These sample codes are not a stand-alone codes.
The users should prepare own field data.

<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1Yo4V7NE28CHHnRZshqc_x_f-3NSJwUNz" width="900" height="300" alt="Flow fields by DNS and ML-ROM"/>
</p>
<p align="center">
    Flow fields computed by DNS (upper line) and predicted by ML-ROM (lower line). This figure shows velocity u, v and pressure p from left. Copyright © 2020 by the Springer.
</p>

# Information  
Author: Kazuto Hasegawa ([Keio University][fukagatalab], Politecnico di Milano)

This repository consists  
1. Multi-Scale_CNN-AE.py (to create Multi-scale CNN-AE)
2. LSTM_with_shape.py (to create LSTM model)

For citations, please use the reference below:
> K. Hasegawa, K. Fukami, T. Murata, and K. Fukagata,  
> "Machine-learning-based reduced-order modeling for unsteady flows around bluff bodies of various shapes,"  
> [Theor. Comput. Fluid Dyn. 34, 367-383 (2020)][thesis].  

Kazuto Hasegawa provides no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission.
The code is written for educational clarity and not for speed.

# Requirements
    * Python 3.x  
    * keras
    * tensorflow
    * sklearn
    * numpy
    * pandas
    * tqdm

# Directory structure
    ML-ROM_Various_Shapes  ── CNN_autoencoder/
                           ├─ data ─── CNNAE ─── data_001.pickle ~ data080.pickle
                           │        │         └─ Test_data/data_001.pickle ~ data020.pickle
                           │        └─ LSTM ─── Dataset/
                           │                 └─ Flags/
                           ├─ .gitignore
                           ├─ LSTM_with_shape.py
                           ├─ Multi-Scale_CNN-AE.py
                           └─ README.md




[thesis]: https://link.springer.com/article/10.1007/s00162-020-00528-w 
[airxiv]: https://arxiv.org/abs/2003.07548
[fukagatalab]: http://kflab.jp/en/index.php?top
