# deletion efficient kmeans


Welcome to the world of deletion efficient AI systems.

This repository contains a research prototype of two provably deletion efficient k-means algorithms, proposed by `tginart` et al. in a forthcoming publication at NeurIPS 2019.

If you use this prototype for research, please reference the original paper [Making AI forget you: Data deletion in machine learning](https://arxiv.org/abs/1907.05012):
```
@inproceedings{ginart2019making,
  title={Making AI forget you: Data deletion in machine learning},
  author={Ginart, Antonio and Guan, Melody and Valiant, Gregory and Zou, James Y},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3513--3526},
  year={2019}
}
```

Please see `demo.ipynb` for a tutorial on using the code. The source code can be found in `del_eff_kmeans.py`. Once you've gone through the demo, you can try out our algorithms on real datasets! See [datasets](https://drive.google.com/open?id=1LqazOJuH3uOgFxHtBodwon6htEE2Wq13) for the preprocessed data used in the paper. All of the data is publicly available as well (see paper for references).
