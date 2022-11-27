# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

### Neural Networks
Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text or time series, must be translated.

### Regression model
A regression model provides a function that describes the relationship between one or more independent variables and a response, dependent, or target variable. For example, the relationship between height and weight may be described by a linear regression mode.
## Neural Network Model

![ALT](nn_arc.jpg "nn_arc.jpg")

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```python
import pandas as pd
from google.colab import auth
import gspread
from google.auth import default

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

auth.aut
worksheet = gc.open('firstdataset').sheet1
rows = worksheet.get_all_values()


df = pd.DataFrame(rows[1:], columns=rows[0])
df.head(n=10)

df.dtypes

df = df.astype({'X':'float'})
df = df.astype({'Y':'float'})

df.dtypes

X = df[['X']].values
Y = df[['Y']].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=50)
X_test.shape

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_scaled


ai_brain = Sequential([
    Dense(2,activation = 'relu'),
    Dense(1,activation = 'relu')
])
ai_brain.compile(optimizer = 'rmsprop',loss = 'mse')
ai_brain.fit(x = X_train_scaled,y = Y_train,epochs = 20000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test
X_test_scaled = scaler.transform(X_test)
X_test_scaled
ai_brain.evaluate(X_test_scaled,Y_test)


input = [[120]]
input_scaled = scaler.transform(input)
input_scaled.shape
input_scaled
ai_brain.predict(input_scaled)henticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


worksheet = gc.open('firstdataset').sheet1
rows = worksheet.get_all_values()


df = pd.DataFrame(rows[1:], columns=rows[0])
df.head(n=10)

df.dtypes

df = df.astype({'X':'float'})
df = df.astype({'Y':'float'})

df.dtypes

X = df[['X']].values
Y = df[['Y']].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=50)
X_test.shape

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_scaled


ai_brain = Sequential([
    Dense(2,activation = 'relu'),
    Dense(1,activation = 'relu')
])
ai_brain.compile(optimizer = 'rmsprop',loss = 'mse')
ai_brain.fit(x = X_train_scaled,y = Y_train,epochs = 20000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test
X_test_scaled = scaler.transform(X_test)
X_test_scaled
ai_brain.evaluate(X_test_scaled,Y_test)


input = [[120]]
input_scaled = scaler.transform(input)
input_scaled.shape
input_scaled
ai_brain.predict(input_scaled)
```

## Dataset Information

![ALT](dataset.png "dataset.png")

## OUTPUT

### Training Loss Vs Iteration Plot

![ALT](plot.png "plot.png")

### Test Data Root Mean Squared Error

0.05628426373004913

### New Sample Data Prediction

![ALT](sample_output.png "sample_output.png")

## RESULT

Succesfully created and trained a neural network regression model for the given dataset.

