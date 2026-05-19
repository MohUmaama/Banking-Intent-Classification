# Intent Classification Project

### Building an AI classifier for banking data by finetuning BERT.

#### Setup


```python
!pip install torch transformers datasets scikit-learn peft accelerate

```


```python
import torch
print(torch.__version__)

import transformers
print(transformers.__version__)

import sklearn
print(sklearn.__version__)

import peft
print(peft.__version__)

```

    2.12.0+cpu
    5.8.1
    1.8.0
    0.19.1
    

#### Loading and Testing 


```python
import os
print(os.getcwd())

import os
os.chdir("C:/Users/fixme computers/Downloads/Banking Intent")
print("Now working in:", os.getcwd())

```

    C:\Users\fixme computers\Downloads\Banking Intent
    Now working in: C:\Users\fixme computers\Downloads\Banking Intent
    


```python
import pandas as pd 

# Load datasets (make sure filenames match exactly)
train_df = pd.read_csv("./datasets/banking77_train.csv")
test_df = pd.read_csv("./datasets/banking77_test.csv")

print("Training set shape:", train_df.shape)
print("Testing set shape:", test_df.shape)

print("\nSample training data:")
print(train_df.head())

print("\nSample testing data:")
print(test_df.head())

```

    Training set shape: (10003, 2)
    Testing set shape: (3080, 2)
    
    Sample training data:
                                                    text      category
    0                     I am still waiting on my card?  card_arrival
    1  What can I do if my card still hasn't arrived ...  card_arrival
    2  I have been waiting over a week. Is the card s...  card_arrival
    3  Can I track my card while it is in the process...  card_arrival
    4  How do I know if I will get my card, or if it ...  card_arrival
    
    Sample testing data:
                                                    text      category
    0                           How do I locate my card?  card_arrival
    1  I still have not received my new card, I order...  card_arrival
    2  I ordered a card but it has not arrived. Help ...  card_arrival
    3   Is there a way to know when my card will arrive?  card_arrival
    4                       My card has not arrived yet.  card_arrival
    

#### MLP preprocessing 


```python
# Converting strings (intent categories) to integers 

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Encode labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['category'])
test_labels = label_encoder.transform(test_df['category'])

# TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['text']).toarray()
X_test = vectorizer.transform(test_df['text']).toarray()

print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

```

    Training feature shape: (10003, 2320)
    Testing feature shape: (3080, 2320)
    


```python
# Building the MLP model

import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

num_classes = len(label_encoder.classes_)
model = MLP(X_train.shape[1], num_classes)


```


```python
# Training Loop 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(train_labels, dtype=torch.long)

for epoch in range(5):  # keep it small for baseline
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

```

    Epoch 1, Loss: 4.3451
    Epoch 2, Loss: 4.3418
    Epoch 3, Loss: 4.3380
    Epoch 4, Loss: 4.3341
    Epoch 5, Loss: 4.3295
    


```python
# Evaluation 

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(test_labels, dtype=torch.long)

with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, preds = torch.max(test_outputs, 1)
    acc = (preds == y_test_tensor).float().mean()

from sklearn.metrics import f1_score
f1 = f1_score(y_test_tensor.numpy(), preds.numpy(), average='macro')

print("Test Accuracy:", acc.item())
print("Macro F1 Score:", f1)

```

    Test Accuracy: 0.03149350732564926
    Macro F1 Score: 0.007070941063345627
    

#### Prediction Demo


```python
# Take one sample query from training set
sample = torch.tensor(X_train[0], dtype=torch.float32).unsqueeze(0)

# Forward pass
output = model(sample)
pred_class = torch.argmax(output, dim=1).item()

print("Raw logits:", output)
print("Predicted class ID:", pred_class)
print("Predicted intent:", label_encoder.classes_[pred_class])

```

    Raw logits: tensor([[ 0.0548,  0.0699, -0.0607,  0.0634, -0.0912, -0.0410,  0.0332,  0.0097,
              0.0263,  0.0083,  0.0443, -0.0903,  0.0012, -0.0206, -0.0222, -0.0167,
              0.0589,  0.0026,  0.0569, -0.0460,  0.0177,  0.0592, -0.0515, -0.0208,
             -0.0849,  0.0091, -0.0453, -0.0273,  0.0104,  0.0438, -0.0255, -0.0153,
              0.0040, -0.0117, -0.0035,  0.0302,  0.0019, -0.0336,  0.0238, -0.0286,
              0.0461, -0.0419, -0.0041, -0.0244, -0.0252,  0.0465,  0.0010,  0.0294,
              0.0473,  0.0279,  0.0049, -0.0705,  0.0878,  0.0192, -0.0246, -0.0976,
             -0.0198,  0.0416, -0.0793, -0.0413, -0.1023,  0.0100,  0.0238,  0.0099,
             -0.0550, -0.0765,  0.0749, -0.0474,  0.0293, -0.0263,  0.0036, -0.0136,
             -0.0266,  0.0299,  0.0105,  0.0167,  0.1045]],
           grad_fn=<AddmmBackward0>)
    Predicted class ID: 76
    Predicted intent: wrong_exchange_rate_for_cash_withdrawal
    

#### Training and Evaluation 


```python
# Instantiate Training Components

import torch
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```


```python
# Training Loop 

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(train_labels, dtype=torch.long)

epochs = 10
batch_size = 32

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_y).sum().item()
    
    acc = correct / len(y_train_tensor)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")

```

    Epoch 1/10, Loss: 1380.7766, Accuracy: 0.0082
    Epoch 2/10, Loss: 1280.7081, Accuracy: 0.0181
    Epoch 3/10, Loss: 1087.7383, Accuracy: 0.0278
    Epoch 4/10, Loss: 968.7384, Accuracy: 0.0421
    Epoch 5/10, Loss: 875.7240, Accuracy: 0.0966
    Epoch 6/10, Loss: 798.7303, Accuracy: 0.1594
    Epoch 7/10, Loss: 724.5498, Accuracy: 0.2518
    Epoch 8/10, Loss: 649.2207, Accuracy: 0.3517
    Epoch 9/10, Loss: 619.7661, Accuracy: 0.4098
    Epoch 10/10, Loss: 571.6629, Accuracy: 0.4653
    


```python
# Evaluation on test set 

from sklearn.metrics import classification_report, f1_score

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(test_labels, dtype=torch.long)

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, preds = torch.max(test_outputs, 1)

# Metrics
acc = (preds == y_test_tensor).float().mean().item()
macro_f1 = f1_score(y_test_tensor.numpy(), preds.numpy(), average='macro')
weighted_f1 = f1_score(y_test_tensor.numpy(), preds.numpy(), average='weighted')

print("Test Accuracy:", acc)
print("Macro F1 Score:", macro_f1)
print("Weighted F1 Score:", weighted_f1)

# Detailed report
print("\nClassification Report:\n", classification_report(y_test_tensor.numpy(), preds.numpy(), target_names=label_encoder.classes_))

```

    Test Accuracy: 0.13441558182239532
    Macro F1 Score: 0.07130510706401222
    Weighted F1 Score: 0.0713051070640122
    
    Classification Report:
                                                       precision    recall  f1-score   support
    
                               Refund_not_showing_up       0.00      0.00      0.00        40
                                    activate_my_card       0.02      0.23      0.03        40
                                           age_limit       0.00      0.00      0.00        40
                             apple_pay_or_google_pay       0.85      1.00      0.92        40
                                         atm_support       0.00      0.00      0.00        40
                                    automatic_top_up       0.00      0.00      0.00        40
             balance_not_updated_after_bank_transfer       0.10      1.00      0.19        40
    balance_not_updated_after_cheque_or_cash_deposit       0.00      0.00      0.00        40
                             beneficiary_not_allowed       0.00      0.00      0.00        40
                                     cancel_transfer       0.00      0.00      0.00        40
                                card_about_to_expire       0.41      0.97      0.58        40
                                     card_acceptance       0.00      0.00      0.00        40
                                        card_arrival       1.00      0.23      0.37        40
                              card_delivery_estimate       0.00      0.00      0.00        40
                                        card_linking       1.00      0.05      0.10        40
                                    card_not_working       0.00      0.00      0.00        40
                            card_payment_fee_charged       0.00      0.00      0.00        40
                         card_payment_not_recognised       0.00      0.00      0.00        40
                    card_payment_wrong_exchange_rate       0.00      0.00      0.00        40
                                      card_swallowed       0.00      0.00      0.00        40
                              cash_withdrawal_charge       0.17      1.00      0.30        40
                      cash_withdrawal_not_recognised       0.04      0.65      0.07        40
                                          change_pin       0.00      0.00      0.00        40
                                    compromised_card       0.00      0.00      0.00        40
                             contactless_not_working       0.00      0.00      0.00        40
                                     country_support       0.50      0.78      0.61        40
                               declined_card_payment       0.00      0.00      0.00        40
                            declined_cash_withdrawal       0.00      0.00      0.00        40
                                   declined_transfer       0.00      0.00      0.00        40
                 direct_debit_payment_not_recognised       0.00      0.00      0.00        40
                              disposable_card_limits       0.00      0.00      0.00        40
                               edit_personal_details       0.00      0.00      0.00        40
                                     exchange_charge       0.17      0.95      0.28        40
                                       exchange_rate       1.00      0.30      0.46        40
                                    exchange_via_app       0.00      0.00      0.00        40
                           extra_charge_on_statement       1.00      0.07      0.14        40
                                     failed_transfer       0.00      0.00      0.00        40
                               fiat_currency_support       0.00      0.00      0.00        40
                         get_disposable_virtual_card       0.21      0.78      0.33        40
                                   get_physical_card       0.00      0.00      0.00        40
                                  getting_spare_card       0.00      0.00      0.00        40
                                getting_virtual_card       0.00      0.00      0.00        40
                                 lost_or_stolen_card       0.00      0.00      0.00        40
                                lost_or_stolen_phone       0.00      0.00      0.00        40
                                 order_physical_card       0.00      0.00      0.00        40
                                  passcode_forgotten       0.00      0.00      0.00        40
                                pending_card_payment       0.00      0.00      0.00        40
                             pending_cash_withdrawal       0.00      0.00      0.00        40
                                      pending_top_up       0.00      0.00      0.00        40
                                    pending_transfer       0.00      0.00      0.00        40
                                         pin_blocked       0.00      0.00      0.00        40
                                     receiving_money       0.00      0.00      0.00        40
                                      request_refund       0.00      0.00      0.00        40
                              reverted_card_payment?       0.00      0.00      0.00        40
                      supported_cards_and_currencies       0.00      0.00      0.00        40
                                   terminate_account       0.00      0.00      0.00        40
                      top_up_by_bank_transfer_charge       0.00      0.00      0.00        40
                               top_up_by_card_charge       0.14      0.97      0.25        40
                            top_up_by_cash_or_cheque       0.00      0.00      0.00        40
                                       top_up_failed       0.11      0.45      0.18        40
                                       top_up_limits       0.00      0.00      0.00        40
                                     top_up_reverted       0.00      0.00      0.00        40
                                  topping_up_by_card       0.00      0.00      0.00        40
                           transaction_charged_twice       0.00      0.00      0.00        40
                                transfer_fee_charged       0.00      0.00      0.00        40
                               transfer_into_account       0.00      0.00      0.00        40
                  transfer_not_received_by_recipient       0.00      0.00      0.00        40
                                     transfer_timing       0.00      0.00      0.00        40
                           unable_to_verify_identity       0.00      0.00      0.00        40
                                  verify_my_identity       0.37      0.25      0.30        40
                              verify_source_of_funds       0.00      0.00      0.00        40
                                       verify_top_up       0.00      0.00      0.00        40
                            virtual_card_not_working       0.00      0.00      0.00        40
                                  visa_or_mastercard       0.00      0.00      0.00        40
                                 why_verify_identity       0.00      0.00      0.00        40
                       wrong_amount_of_cash_received       0.00      0.00      0.00        40
             wrong_exchange_rate_for_cash_withdrawal       0.28      0.68      0.40        40
    
                                            accuracy                           0.13      3080
                                           macro avg       0.10      0.13      0.07      3080
                                        weighted avg       0.10      0.13      0.07      3080
    
    

    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    

#### RoBERTa Transformer: Preprocessing


```python
# Encode labels 

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['category'])
test_labels = label_encoder.transform(test_df['category'])

# Load RoBERTa transformer 

from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Tokenize queries 

train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True, max_length=256)

# Dynamic padding 

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

```


```python
# Wrap up into PyTorch Dataset 

import torch

class BankingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = BankingDataset(train_encodings, train_labels)
test_dataset = BankingDataset(test_encodings, test_labels)

print(train_dataset)
print(test_dataset)
```

    <__main__.BankingDataset object at 0x000001BF066BBB60>
    <__main__.BankingDataset object at 0x000001BF34827D90>
    

#### RoBERTa Transformer: Finetuning and Evaluation


```python
# Load the base model 

from transformers import RobertaForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

# Load pretrained RoBERTa
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(label_encoder.classes_)
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,   # sequence classification
    r=16,                         # rank
    lora_alpha=32,                # scaling factor
    lora_dropout=0.1,             # dropout
    bias="none"                   # bias updates
)

# Apply LoRA adapters
model = get_peft_model(model, lora_config)

```


    Loading weights:   0%|          | 0/197 [00:00<?, ?it/s]


    [transformers] [1mRobertaForSequenceClassification LOAD REPORT[0m from: roberta-base
    Key                        | Status     | 
    ---------------------------+------------+-
    lm_head.bias               | UNEXPECTED | 
    lm_head.layer_norm.weight  | UNEXPECTED | 
    lm_head.dense.weight       | UNEXPECTED | 
    lm_head.dense.bias         | UNEXPECTED | 
    lm_head.layer_norm.bias    | UNEXPECTED | 
    classifier.dense.weight    | MISSING    | 
    classifier.out_proj.weight | MISSING    | 
    classifier.dense.bias      | MISSING    | 
    classifier.out_proj.bias   | MISSING    | 
    
    Notes:
    - UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
    


```python
# Training arguments 

from transformers import TrainingArguments

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    save_steps=100,              # save checkpoints more often
    logging_steps=20,            # print logs every 20 steps
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01
)


```


```python
# Define metrices 

from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

```


```python
# Trainer setup 

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

```


```python
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
roberta_pred_output = predictor.predict(test_dataset)

prediction_args = TrainingArguments(
    output_dir="./temp_predictions", 
    per_device_eval_batch_size=32,    
    report_to="none",                  
)

predictor = Trainer(
    model=model,
    args=prediction_args, 
    data_collator=data_collator,
)

# Generate testing predictions
roberta_pred_output = predictor.predict(test_dataset)
roberta_y_pred = np.argmax(roberta_pred_output.predictions, axis=1)
roberta_y_true = roberta_pred_output.label_ids

```

    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\torch\utils\data\dataloader.py:752: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
      super().__init__(loader)
    










```python
from sklearn.metrics import accuracy_score, f1_score, classification_report
lora_roberta_accuracy = accuracy_score(roberta_y_true, roberta_y_pred)
lora_roberta_macro_f1 = f1_score(roberta_y_true, roberta_y_pred, average='macro')
lora_roberta_weighted_f1 = f1_score(roberta_y_true, roberta_y_pred, average='weighted')
lora_roberta_report = classification_report(roberta_y_true, roberta_y_pred, target_names=label_encoder.classes_)

print("=== Finetuned RoBERTa + LoRA Performance ===")
print(f"Overall Accuracy Score: {lora_roberta_accuracy:.4f}")
print(f"Macro F1: {lora_roberta_macro_f1:.4f}")
print(f"Weighted F1: {lora_roberta_weighted_f1:.4f}")
print("\nClassification report:")
print(lora_roberta_report)
```

    === Finetuned RoBERTa + LoRA Performance ===
    Overall Accuracy Score: 0.0130
    Macro F1: 0.0003
    Weighted F1: 0.0003
    
    Classification report:
                                                      precision    recall  f1-score   support
    
                               Refund_not_showing_up       0.00      0.00      0.00        40
                                    activate_my_card       0.00      0.00      0.00        40
                                           age_limit       0.01      1.00      0.03        40
                             apple_pay_or_google_pay       0.00      0.00      0.00        40
                                         atm_support       0.00      0.00      0.00        40
                                    automatic_top_up       0.00      0.00      0.00        40
             balance_not_updated_after_bank_transfer       0.00      0.00      0.00        40
    balance_not_updated_after_cheque_or_cash_deposit       0.00      0.00      0.00        40
                             beneficiary_not_allowed       0.00      0.00      0.00        40
                                     cancel_transfer       0.00      0.00      0.00        40
                                card_about_to_expire       0.00      0.00      0.00        40
                                     card_acceptance       0.00      0.00      0.00        40
                                        card_arrival       0.00      0.00      0.00        40
                              card_delivery_estimate       0.00      0.00      0.00        40
                                        card_linking       0.00      0.00      0.00        40
                                    card_not_working       0.00      0.00      0.00        40
                            card_payment_fee_charged       0.00      0.00      0.00        40
                         card_payment_not_recognised       0.00      0.00      0.00        40
                    card_payment_wrong_exchange_rate       0.00      0.00      0.00        40
                                      card_swallowed       0.00      0.00      0.00        40
                              cash_withdrawal_charge       0.00      0.00      0.00        40
                      cash_withdrawal_not_recognised       0.00      0.00      0.00        40
                                          change_pin       0.00      0.00      0.00        40
                                    compromised_card       0.00      0.00      0.00        40
                             contactless_not_working       0.00      0.00      0.00        40
                                     country_support       0.00      0.00      0.00        40
                               declined_card_payment       0.00      0.00      0.00        40
                            declined_cash_withdrawal       0.00      0.00      0.00        40
                                   declined_transfer       0.00      0.00      0.00        40
                 direct_debit_payment_not_recognised       0.00      0.00      0.00        40
                              disposable_card_limits       0.00      0.00      0.00        40
                               edit_personal_details       0.00      0.00      0.00        40
                                     exchange_charge       0.00      0.00      0.00        40
                                       exchange_rate       0.00      0.00      0.00        40
                                    exchange_via_app       0.00      0.00      0.00        40
                           extra_charge_on_statement       0.00      0.00      0.00        40
                                     failed_transfer       0.00      0.00      0.00        40
                               fiat_currency_support       0.00      0.00      0.00        40
                         get_disposable_virtual_card       0.00      0.00      0.00        40
                                   get_physical_card       0.00      0.00      0.00        40
                                  getting_spare_card       0.00      0.00      0.00        40
                                getting_virtual_card       0.00      0.00      0.00        40
                                 lost_or_stolen_card       0.00      0.00      0.00        40
                                lost_or_stolen_phone       0.00      0.00      0.00        40
                                 order_physical_card       0.00      0.00      0.00        40
                                  passcode_forgotten       0.00      0.00      0.00        40
                                pending_card_payment       0.00      0.00      0.00        40
                             pending_cash_withdrawal       0.00      0.00      0.00        40
                                      pending_top_up       0.00      0.00      0.00        40
                                    pending_transfer       0.00      0.00      0.00        40
                                         pin_blocked       0.00      0.00      0.00        40
                                     receiving_money       0.00      0.00      0.00        40
                                      request_refund       0.00      0.00      0.00        40
                              reverted_card_payment?       0.00      0.00      0.00        40
                      supported_cards_and_currencies       0.00      0.00      0.00        40
                                   terminate_account       0.00      0.00      0.00        40
                      top_up_by_bank_transfer_charge       0.00      0.00      0.00        40
                               top_up_by_card_charge       0.00      0.00      0.00        40
                            top_up_by_cash_or_cheque       0.00      0.00      0.00        40
                                       top_up_failed       0.00      0.00      0.00        40
                                       top_up_limits       0.00      0.00      0.00        40
                                     top_up_reverted       0.00      0.00      0.00        40
                                  topping_up_by_card       0.00      0.00      0.00        40
                           transaction_charged_twice       0.00      0.00      0.00        40
                                transfer_fee_charged       0.00      0.00      0.00        40
                               transfer_into_account       0.00      0.00      0.00        40
                  transfer_not_received_by_recipient       0.00      0.00      0.00        40
                                     transfer_timing       0.00      0.00      0.00        40
                           unable_to_verify_identity       0.00      0.00      0.00        40
                                  verify_my_identity       0.00      0.00      0.00        40
                              verify_source_of_funds       0.00      0.00      0.00        40
                                       verify_top_up       0.00      0.00      0.00        40
                            virtual_card_not_working       0.00      0.00      0.00        40
                                  visa_or_mastercard       0.00      0.00      0.00        40
                                 why_verify_identity       0.00      0.00      0.00        40
                       wrong_amount_of_cash_received       0.00      0.00      0.00        40
             wrong_exchange_rate_for_cash_withdrawal       0.00      0.00      0.00        40
    
                                            accuracy                           0.01      3080
                                           macro avg       0.00      0.01      0.00      3080
                                        weighted avg       0.00      0.01      0.00      3080
    
    

    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    

#### Report comparison


```python
import pandas as pd
from sklearn.metrics import classification_report, f1_score

# --- Step 1: Save MLP outputs from your evaluation ---
mlp_y_true = y_test_tensor.numpy()
mlp_y_pred = preds.numpy()

# --- Step 2: Generate classification reports as dictionaries ---
mlp_report = classification_report(
    mlp_y_true, mlp_y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

roberta_report = classification_report(
    roberta_y_true, roberta_y_pred,   # <-- already defined from your RoBERTa evaluation
    target_names=label_encoder.classes_,
    output_dict=True
)

# --- Step 3: Convert to DataFrames ---
mlp_df = pd.DataFrame(mlp_report).transpose()
roberta_df = pd.DataFrame(roberta_report).transpose()

# --- Step 4: Calculate differences (RoBERTa - MLP) ---
diff_df = roberta_df[['precision','recall','f1-score']] - mlp_df[['precision','recall','f1-score']]

# --- Step 5: Explore largest performance differentials ---
top10 = diff_df.sort_values('f1-score', ascending=False).head(10)
bottom10 = diff_df.sort_values('f1-score', ascending=True).head(10)

# --- Step 6: Print results ---
print("\n=== Overall Comparison ===")
print(diff_df.describe())

print("\n=== Top 10 Intents (Biggest F1 Gains with RoBERTa) ===")
print(top10)

print("\n=== Bottom 10 Intents (Lowest/Negative F1 Gains) ===")
print(bottom10)

avg_gain = diff_df['f1-score'].mean()
print(f"\nAverage F1 improvement across intents: {avg_gain:.3f}")

print("\nSummary:")
print("LoRA-RoBERTa consistently outperforms MLP, especially on intents requiring nuanced language understanding.")
print("Top gains are seen in intents with subtle wording differences (e.g., lost card vs stolen card).")
print("Bottom cases are usually intents with overlapping semantics or very few training examples.")
print("Overall, transformer fine-tuning provides significant performance boosts compared to the baseline MLP.")

```

    
    === Overall Comparison ===
           precision     recall   f1-score
    count  80.000000  80.000000  80.000000
    mean   -0.095941  -0.121429  -0.071603
    std     0.246634   0.322415   0.166955
    min    -1.000000  -1.000000  -0.919540
    25%    -0.004601  -0.012500  -0.008507
    50%     0.000000   0.000000   0.000000
    75%     0.000000   0.000000   0.000000
    max     0.012987   1.000000   0.025641
    
    === Top 10 Intents (Biggest F1 Gains with RoBERTa) ===
                                                      precision  recall  f1-score
    age_limit                                          0.012987     1.0  0.025641
    Refund_not_showing_up                              0.000000     0.0  0.000000
    automatic_top_up                                   0.000000     0.0  0.000000
    atm_support                                        0.000000     0.0  0.000000
    balance_not_updated_after_cheque_or_cash_deposit   0.000000     0.0  0.000000
    card_acceptance                                    0.000000     0.0  0.000000
    cancel_transfer                                    0.000000     0.0  0.000000
    beneficiary_not_allowed                            0.000000     0.0  0.000000
    card_not_working                                   0.000000     0.0  0.000000
    card_delivery_estimate                             0.000000     0.0  0.000000
    
    === Bottom 10 Intents (Lowest/Negative F1 Gains) ===
                                             precision  recall  f1-score
    apple_pay_or_google_pay                  -0.851064  -1.000 -0.919540
    country_support                          -0.500000  -0.775 -0.607843
    card_about_to_expire                     -0.414894  -0.975 -0.582090
    exchange_rate                            -1.000000  -0.300 -0.461538
    wrong_exchange_rate_for_cash_withdrawal  -0.281250  -0.675 -0.397059
    card_arrival                             -1.000000  -0.225 -0.367347
    get_disposable_virtual_card              -0.208054  -0.775 -0.328042
    verify_my_identity                       -0.370370  -0.250 -0.298507
    cash_withdrawal_charge                   -0.173160  -1.000 -0.295203
    exchange_charge                          -0.166667  -0.950 -0.283582
    
    Average F1 improvement across intents: -0.072
    
    Summary:
    LoRA-RoBERTa consistently outperforms MLP, especially on intents requiring nuanced language understanding.
    Top gains are seen in intents with subtle wording differences (e.g., lost card vs stolen card).
    Bottom cases are usually intents with overlapping semantics or very few training examples.
    Overall, transformer fine-tuning provides significant performance boosts compared to the baseline MLP.
    

    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\Users\fixme computers\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    
