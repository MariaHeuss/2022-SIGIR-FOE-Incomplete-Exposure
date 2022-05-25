# Fairness of Exposure in Light of Incomplete Exposure Estimation 

This repository contains the code used for the experiments in 
"Fairness of Exposure in Light of Incomplete Exposure Estimation", 
which will be published at SIGIR 2022.

## Citation
If you use this code to produce results for your scientific publication, or if you share a copy or fork, 
please refer to our SIGIR 2022 paper:

```
@inproceedings{heuss-2022-felix,
  Author = {Heuss, Maria and Sarvi, Fatemeh and de Rijke, Maarten},
  Booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR`22)},
  Organization = {ACM},
  Title = {Fairness of Exposure in Light of Incomplete Exposure Estimation},
  Year = {2022}
}
```

## Licence
This repository is published under the terms of the GNU General Public License version 3. 
Lor more information, see the file LICENSE.

```
Fairness of Exposure in Light of Incomplete Exposure Estimation 
Copyright (C) 2022 Maria Heuss

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
```

## Usage
Packages used in this repository can be found in requirements.txt 

To run the experiments that are recorded in the paper run 'run_felix'. 
This will new files in the '\results\' folder, one for each experiment 
on top-k and full rankings on both of the TREC19 and TREC20 fair ranking
datasets. Also, for each of the experiments on simulated data there will 
be a new file with the results as well as a figure in the '\results\figures'
folder. Additionally the tables with the results will be sent to stdout. 


## Credits 
This project is build on top of
- https://github.com/MilkaLichtblau/BA_Laura,
which we used as basis for big parts of the code and 
- https://github.com/arezooSarvi/OMIT_Fair_ranking, which we used to copy 
preprocessed data in the data folder and utilized some of the evaluation, 
  the OMIT method as baseline.

Furthermore in our implementation we use and adjust parts of 
- https://github.com/fullflu/learning-to-rank
- https://github.com/jfinkels/birkhoff, which is published under the e 
  GNU General Public License
- https://networkx.org, which is published under the 3-clause BSD licence 
- https://github.com/HarrieO/2021-SIGIR-plackett-luce, which is published
  under the MIT licence.
  