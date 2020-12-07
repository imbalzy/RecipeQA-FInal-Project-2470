import numpy as np
import tensorflow as tf
from preprocessrecipeqa import *

class Model(tf.keras.Model):
  def __init__(self, l_embed):
    super(Model, self).__init__()
    self.image_conv1 = tf.keras.layers.Conv2D(filters = 4, kernel_size = 3, strides=(2, 2), padding='valid')
    self.image_conv2 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, strides=(2, 2), padding='valid')
    self.image_conv3 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, strides=(2, 2), padding='valid')
    self.image_dense = tf.keras.layers.Dense(units = 100)

    self.word_embedding = l_embed
    self.text_dense = tf.keras.layers.Dense(units = 100)

    self.text_image_embedding1 = tf.keras.layers.Dense(units = 200, activation = "relu")
    self.text_image_embedding2 = tf.keras.layers.Dense(units = 100)

    self.class_dense1 = tf.keras.layers.Dense(units = 50, activation = "relu")
    self.class_dense2 = tf.keras.layers.Dense(units = 20, activation = "relu")
    self.class_dense3 = tf.keras.layers.Dense(units = 4)
    self.optimizer = tf.keras.optimizers.Adam(1e-4)
    self.gru = tf.keras.layers.GRU(100,return_state=True,return_sequences=True)

  def call(self, Xs):
    textlist = []
    choicelist = []
    text = []
    '''
    for recipe in Xs:
      text = []
      for step in recipe['context']:
        text += step['body']
      text = tf.reduce_mean([l_embed(tf.convert_to_tensor(item)) for item in text],axis=0)
      textlist.append(text)
      choicelist.append(recipe['choice_list'])
    textlist = tf.convert_to_tensor(textlist)
    '''
    for step in Xs[0]['context']:
      text += step['body']
    text = self.word_embedding(tf.convert_to_tensor(text))
#    print(tf.expand_dims(text,axis=0).shape)
    _,textlist = self.gru(tf.expand_dims(text,axis=0), None)
#    print(textlist.shape)
    choicelist.append(Xs[0]['choice_list'])
    textlist = tf.convert_to_tensor(textlist)
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

    text_embedding = self.text_dense(textlist)

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
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))
    
batch_size = 1
(train_iter1, train_iter2, train_iter3, train_iter4), (test_iter1, test_iter2, test_iter3, test_iter4), (val_iter1, val_iter2, val_iter3, val_iter4), embedding_index, word_index = preprocess(batch_size)
l_embed = get_embedding_layer(word_index, embedding_index)

def train(model, iter):
  for Xs, Ys in iter:
    with tf.GradientTape() as tape:
      logits = model(Xs)
      loss = model.loss(logits, Ys)
      print(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
def test(model, iter):
  n = 0
  m = 0
  for Xs, Ys in iter:
    n += len(Xs)
    probs = model(Xs)
    m += sum(np.argmax(probs,-1)==Ys)
  print(m/n)
  return m/n
    
model = Model(l_embed)
train(model, train_iter3)
test(model, test_iter3)
