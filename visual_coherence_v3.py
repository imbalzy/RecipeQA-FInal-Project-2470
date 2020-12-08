import numpy as np
import tensorflow as tf
from preprocessrecipeqa import *
import transformer_funcs as transformer

from attenvis import AttentionVis

class Model(tf.keras.Model):
  def __init__(self, l_embed):
    super(Model, self).__init__()
    self.window_sz = 500
    self.embedding_size = 100
    self.image_conv1 = tf.keras.layers.Conv2D(filters = 4, kernel_size = 3, strides=(2, 2), padding='valid')
    self.image_conv2 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, strides=(2, 2), padding='valid')
    self.image_conv3 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, strides=(2, 2), padding='valid')
    self.image_dense = tf.keras.layers.Dense(units = 100)

    self.word_embedding = l_embed
    self.text_step_conv1 = tf.keras.layers.Conv1D(filters = 16, kernel_size = 5)
    self.text_step_conv2 = tf.keras.layers.Conv1D(filters = 32, kernel_size = 5)
    self.text_step_conv3 = tf.keras.layers.Conv1D(filters = 32, kernel_size = 5)
    self.text_step_maxpool = tf.keras.layers.MaxPool1D(pool_size = 3)
    self.text_step_dense = tf.keras.layers.Dense(units = 100)

    self.gru = tf.keras.layers.GRU(200, return_sequences=True, return_state=True)

    self.text_image_embedding1 = tf.keras.layers.Dense(units = 200, activation = "relu")
    self.text_image_embedding2 = tf.keras.layers.Dense(units = 100)

    self.class_dense1 = tf.keras.layers.Dense(units = 50, activation = "relu")
    self.class_dense2 = tf.keras.layers.Dense(units = 20, activation = "relu")
    self.class_dense3 = tf.keras.layers.Dense(units = 4)
    self.optimizer = tf.keras.optimizers.Adam(1e-3)

  def call(self, Xs):
    textlist = []
    choicelist = []
    for recipe in Xs:#per data
      text = []
      step_list = []#all step per data
      for step in recipe['context']:#per step
        step_list.append(step['body'])
        #text += step['body']
      #print("call text len: ",len(text))
      #print("call text: ",text)
      #text = tf.reduce_mean([l_embed(tf.convert_to_tensor(item)) for item in text],axis=0)
      textlist.append(steplist)
      choicelist.append(recipe['choice_list'])
    textlist = tf.convert_to_tensor(textlist)
    print("call textlist shape: ", textlist.shape)
    choice_image = tf.convert_to_tensor(choicelist)

    choice_token = self.image_conv1(choice_image[:,0])
    choice_token = self.image_conv2(choice_token)
    choice_token = self.image_conv3(choice_token)
    choice0_embedding = self.image_dense(tf.reshape(choice_token,(choice_token.shape[0],-1)))

    choice_token = self.image_conv1(choice_image[:,1])
    choice_token = self.image_conv2(choice_token)
    choice_token = self.image_conv3(choice_token)
    choice1_embedding = self.image_dense(tf.reshape(choice_token,(choice_token.shape[0],-1)))

    choice_token = self.image_conv1(choice_image[:,2])
    choice_token = self.image_conv2(choice_token)
    choice_token = self.image_conv3(choice_token)
    choice2_embedding = self.image_dense(tf.reshape(choice_token,(choice_token.shape[0],-1)))

    choice_token = self.image_conv1(choice_image[:,3])
    choice_token = self.image_conv2(choice_token)
    choice_token = self.image_conv3(choice_token)
    choice3_embedding = self.image_dense(tf.reshape(choice_token,(choice_token.shape[0],-1)))

    #query word embedding

    #text_embedding = self.text_dense(textlist)
    text_embedding = []
    for all_steps in textlist:#per data
        all_steps_embedding = []
        for step in all_steps:#per step
            step_token = self.text_step_conv1(step)
            step_token = self.text_step_conv2(step_token)
            step_token = self.text_step_conv3(step_token)
            step_token = self.text_step_maxpool(step_token)
            step_token = self.text_step_dense(step_token)
            all_steps_embedding.append(step_token)
        all_steps_embedding = tf.convert_to_tensor(all_steps_embedding)
        output, state = self.gru(all_steps_embedding)
        text_embedding.append(output)

    text_embedding = tf.convert_to_tensor(text_embedding)

    print("model.call, textlist shape: ",textlist.shape)
    print("call, text_embedding shape: ",text_embedding.shape)
    print("------------------------------")

    #create image_word embedding for choice 0
    token = tf.concat([choice0_embedding, text_embedding],axis=-1)
    token = self.text_image_embedding1(token)
    choice0_embedding = self.text_image_embedding2(token)

     #create image_word embedding for choice 1
    token = tf.concat([choice1_embedding, text_embedding],axis=-1)
    token = self.text_image_embedding1(token)
    choice1_embedding = self.text_image_embedding2(token)

     #create image_word embedding for choice 2
    token = tf.concat([choice2_embedding, text_embedding],axis=-1)
    token = self.text_image_embedding1(token)
    choice2_embedding = self.text_image_embedding2(token)

     #create image_word embedding for choice 3
    token = tf.concat([choice3_embedding, text_embedding],axis=-1)
    token = self.text_image_embedding1(token)
    choice3_embedding = self.text_image_embedding2(token)

    token = tf.concat([choice0_embedding, choice1_embedding, choice2_embedding, choice3_embedding],axis=-1)
    token = self.class_dense1(token)
    token = self.class_dense2(token)
    logit = self.class_dense3(token)

    return logit

  def loss(self, logits, labels):
    print("labels: ",labels)
    print("logits: ",logits)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

batch_size = 3
(train_iter1, train_iter2, train_iter3, train_iter4), (test_iter1, test_iter2, test_iter3, test_iter4), (val_iter1, val_iter2, val_iter3, val_iter4), embedding_index, word_index = preprocess(batch_size)
l_embed = get_embedding_layer(word_index, embedding_index)

def train(model, iter):
  for Xs, Ys in iter:
    with tf.GradientTape() as tape:
      #print("size of iter: ",len(iter))
      logits = model(Xs)
      loss = model.loss(logits, Ys)
      print("train, losse:",loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, iter):
  n = 0
  m = 0
  for Xs, Ys in iter:
    print("-----------------------")
    print("testing, Xs:",len(Xs))
    print("testing, n: ",n)
    n += len(Xs)
    probs = model(Xs)
    m += sum(np.argmax(probs,-1)==Ys)
  print("final test accuracy:",m/n)
  return m/n

model = Model(l_embed)
train(model, trian_iter3)
test(model, test_iter3)
