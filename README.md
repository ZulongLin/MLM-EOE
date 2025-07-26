### This repository contains the code presented in the work “MLM-EOE: Automatic Depression Detection via Sentimental Annotation and Multi-Expert Ensemble” by Zulong Lin, Yaowei Wang, Yujue Zhou, Fei Du, and Yun Yang. (IEEE TAFFC2025)

#### You can run the code as follows: 
1. Install all the packages you need in the requirements.txt.
2. Use the feature extractor to perform pre-training on the face dataset and save checkpoint 1.
3. After loading checkpoint 1, fine-tune it on the depression data and save checkpoint 2.
4. Load checkpoint 2 for feature extraction and convert the depression video frame into a time series.
5. Use the code in utils.py to perform multimodal large model emotion annotation to obtain a time series of depression with emotion annotation.
6. Load the depression time series with emotional annotations and run main.py.

#### If you find this repository useful in your research, please cite:
@article{lin2025mlm,
  title={MLM-EOE: Automatic Depression Detection via Sentimental Annotation and Multi-Expert Ensemble},
  author={Lin, Zulong and Wang, Yaowei and Zhou, Yujue and Du, Fei and Yang, Yun},
  journal={IEEE Transactions on Affective Computing},
  year={2025},
  publisher={IEEE}
}


