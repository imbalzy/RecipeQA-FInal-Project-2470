import numpy as np
import tensorflow as tf
from preprocessrecipeqa import *

class Model(tf.keras.Model):
  def __init__(self, l_embed):
    super(Model, self).__init__()
    self.image_conv1 = tf.keras.layers.Conv2D(filters = 4, kernel_size = 3, strides=(2, 2), padding='valid')
    self.image_conv2 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, strides=(2, 2), padding='valid')
    self.image_conv3 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, strides=(2, 2), padding='valid')
    self.image_dense = tf.keras.layers.Dense(units = 100, activation = "relu")

    self.word_embedding = l_embed
    self.text_dense1 = tf.keras.layers.Dense(units = 100, activation = "relu")
    self.text_dense2 = tf.keras.layers.Dense(units = 100, activation = "relu")
    self.qdense = tf.keras.layers.Dense(units = 100, activation = "relu")

    self.text_image_embedding1 = tf.keras.layers.Dense(units = 100, activation = "relu")
    self.text_image_embedding2 = tf.keras.layers.Dense(units = 100, activation = "relu")

    self.class_dense1 = tf.keras.layers.Dense(units = 100, activation = "relu")
    self.class_dense2 = tf.keras.layers.Dense(units = 4)
    self.optimizer = tf.keras.optimizers.Adam(1e-3)

  def call(self, Xs):
    textlist = []
    choicelist = []
    qlist = []
    for recipe in Xs:
      text = []
      for step in recipe['context']:
        text += step['body']
      text = tf.reduce_mean(self.word_embedding(tf.convert_to_tensor(text)),axis=0)
      textlist.append(text)
      choicelist.append(recipe['choice_list'])
      qlist.append(recipe['question'])
    textlist = tf.convert_to_tensor(textlist)

    qlist = tf.convert_to_tensor(qlist)
    q0 = self.image_conv3(self.image_conv2(self.image_conv1(qlist[:,0])))
    q0 = tf.reshape(q0,(q0.shape[0],-1))
    q1 = self.image_conv3(self.image_conv2(self.image_conv1(qlist[:,1])))
    q1 = tf.reshape(q1,(q1.shape[0],-1))
    q2 = self.image_conv3(self.image_conv2(self.image_conv1(qlist[:,2])))
    q2 = tf.reshape(q2,(q2.shape[0],-1))
    qout = self.qdense(tf.concat([q0,q1,q2],axis=-1))

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

    text_embedding = self.text_dense2(self.text_dense1(textlist))

    #create image_word embedding for choice 0
    token = tf.concat([choice0_embedding, text_embedding, qout],axis=-1)
    token = self.text_image_embedding1(token)
    choice0_embedding = self.text_image_embedding2(token)

     #create image_word embedding for choice 1
    token = tf.concat([choice1_embedding, text_embedding, qout],axis=-1)
    token = self.text_image_embedding1(token)
    choice1_embedding = self.text_image_embedding2(token)

     #create image_word embedding for choice 2
    token = tf.concat([choice2_embedding, text_embedding, qout],axis=-1)
    token = self.text_image_embedding1(token)
    choice2_embedding = self.text_image_embedding2(token)

     #create image_word embedding for choice 3
    token = tf.concat([choice3_embedding, text_embedding, qout],axis=-1)
    token = self.text_image_embedding1(token)
    choice3_embedding = self.text_image_embedding2(token)

    token = tf.concat([choice0_embedding, choice1_embedding, choice2_embedding, choice3_embedding],axis=-1)
    token = self.class_dense1(token)
    logit = self.class_dense2(token)

    return logit
    
  def loss(self, logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))
    
batch_size = 50
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
train(model, train_iter2)
test(model, test_iter2)

