# 🏦 Banking Intent 

![AI Engineering](https://img.shields.io/badge/AI%20Engineering-blueviolet)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)

## 👩‍💻 About Me 

Hello 👋

I'm **Mohamed Fahim Umaama**, an undergraduate AI Engineer passionate about Machine Learning, NLP, and intelligent systems.

## 📌 Project Overview 

This project explores intent classification in banking customer queries using two approaches:
- **Baseline MLP** with TF‑IDF features
- **Finetuned RoBERTa (with LoRA adapters)**

The dataset used is **Banking77**, a benchmark dataset with 77 intent categories.


## ⚙️ Setup  
Clone the repo and install dependencies:  
git clone https://github.com/MohUmaama/Banking-Intent-Classification.git  
cd Banking-Intent-Classification  
pip install -r requirements.txt  

📦 Requirements  
- Python 3.x  
- PyTorch, Transformers, scikit‑learn  

## 💻 How to Run  
1. Install requirements: pip install -r requirements.txt  
2. Open the notebooks in VS Code/Jupyter:  
   - banking_classification_BERT.ipynb → MLP baseline  
   - banking_classification_BERT_solution.ipynb → RoBERTa + LoRA  

## 📊 Sample Output  
- MLP baseline → Accuracy: 13.4%, Macro F1: 0.07  
- RoBERTa + LoRA → Accuracy: 1.3%, Macro F1: 0.0003  
- Demo query: "My card is not working" → Predicted intent: card_not_working  

Thanks for checking out my project! 😊  
Moahmed Fahim Umaama (AI Engineer)
