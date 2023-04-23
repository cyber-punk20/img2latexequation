from __future__ import absolute_import
from keras.layers import Input, Dense, Dropout, \
                         RepeatVector, LSTM, concatenate, \
                         Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import RMSprop
from keras import *
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import numpy as np
# from tensorflow.keras.layers import Input, Dense, Dropout, \
#                          RepeatVector, LSTM, concatenate, \
#                          Conv2D, MaxPooling2D, Flatten
# from tensorflow.keras.models import Sequential, Model, model_from_json
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.callbacks import ModelCheckpoint
from constants import *

from preprocessing import *

import os
class BasicModel:
    def __init__(self, input_shape, output_size, output_dir, checkpoint_path, strategy):
        self.model = None
        self.input_shape = input_shape
        self.output_size = output_size
        self.output_path = output_dir
        self.checkpoint_path = checkpoint_path
        self.name = ""
        self.strategy = strategy
        self.voc = Dataset().voc
        self.voc.loadVolcabulary()
        self.voc.create_binary_representation()
    def save(self):
        model_json = self.model.to_json()
        # with open("{}/{}.json".format(self.output_path, self.name), "w") as json_file:
        #     json_file.write(model_json)
        # self.model.save_weights("{}/{}_weights".format(self.output_path, self.name), save_format='tf')
        self.model.save(self.output_path)

    def load(self, name=""):
        print('Loading model')
        with self.strategy.scope():
            output_name = self.name if name == "" else name
            # with open("{}/{}.json".format(self.output_path, output_name), "r") as json_file:
            #     loaded_model_json = json_file.read()
            #     self.model = model_from_json(loaded_model_json)
            #     self.model.load_weights("{}/{}_weights".format(self.output_path, self.name))
            self.model = load_model(self.output_path)

    def load_for_evaluation(self, name=""):
        output_name = self.name if name == "" else name
        # with open("{}/{}.json".format(self.output_path, output_name), "r") as json_file:
        #     loaded_model_json = json_file.read()
        #     self.model = model_from_json(loaded_model_json)
        #     self.model.load_weights("{}/{}_weights".format(self.output_path, self.name))
        self.model = load_model(self.output_path)
    


class pix2equation(BasicModel):
    def __init__(self, input_shape, output_size, output_dir, checkpoint_path, strategy):
        super().__init__(input_shape, output_size, output_dir, checkpoint_path, strategy)
        self.name = "pix2equation"
        
        with self.strategy.scope():
            image_model = Sequential()
            image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
            image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
            image_model.add(MaxPooling2D(pool_size=(2, 2)))
            image_model.add(Dropout(0.25))

            image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
            image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
            image_model.add(MaxPooling2D(pool_size=(2, 2)))
            image_model.add(Dropout(0.25))

            image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
            image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
            image_model.add(MaxPooling2D(pool_size=(2, 2)))
            image_model.add(Dropout(0.25))

            image_model.add(Flatten())
            image_model.add(Dense(1024, activation='relu'))
            image_model.add(Dropout(0.3))
            image_model.add(Dense(1024, activation='relu'))
            image_model.add(Dropout(0.3))

            image_model.add(RepeatVector(CONTEXT_LENGTH))

            visual_input = Input(shape=input_shape)
            encoded_image = image_model(visual_input)

            language_model = Sequential()
            language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
            language_model.add(LSTM(128, return_sequences=True))

            textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
            encoded_text = language_model(textual_input)
            decoder = concatenate([encoded_image, encoded_text])

            decoder = LSTM(512, return_sequences=True)(decoder)
            decoder = LSTM(512, return_sequences=False)(decoder)
            decoder = Dense(output_size, activation='softmax')(decoder)

            self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            self.checkpoint = ModelCheckpoint(checkpoint_path, 
                                              monitor='val_loss', 
                                              verbose=1, 
                                              save_best_only=True, 
                                              mode='min',
                                              save_format='tf')
            optimizer = RMSprop(learning_rate=0.0001, clipvalue=1.0)
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    def compile(self):
        with self.strategy.scope():
            optimizer = RMSprop(learning_rate=0.0001, clipvalue=1.0)
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    def fit(self, train_generator, batch_size, steps_per_epoch, epochs, valid_generator, validation_steps):
        # with self.strategy.scope():
        self.model.fit(train_generator, 
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=valid_generator,
            validation_steps=validation_steps,
            callbacks=[self.checkpoint],
            verbose=1)
        self.save()

    # def fit(self, images, partial_captions, next_words, batch_size, steps_per_epoch, v_images, v_partial_captions, v_next_words, validation_steps, checkpoint):
    #     self.model.fit([images, partial_captions], 
    #                     next_words,
    #                     batch_size=batch_size,
    #                     steps_per_epoch=steps_per_epoch,
    #                     validation_data=([v_images, v_partial_captions], v_next_words),
    #                     validation_steps=validation_steps,
    #                     callbacks=[checkpoint],
    #                     verbose=1)
    #     self.save()

    def predict(self, image, partial_caption):
        # print("partial_caption shape:", partial_caption.shape)
        # print("partial_caption dtype:", partial_caption.dtype)
        return self.model.predict([image, partial_caption], verbose=0)[0]

    def predict_batch(self, images, partial_captions):
        return self.model.predict([images, partial_captions], verbose=1)


    def predict_greedy(self, image, sequence_length):
        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (CONTEXT_LENGTH)
        # current_context.append(self.voc.vocabulary[START_TOKEN])
        current_context = Dataset.sparsify(current_context, self.output_size)
        #  predictions = [START_TOKEN]
        predictions = []
        for i in range(0, sequence_length):
            # print(current_context.shape)
            probas = self.predict(image, np.array([current_context]))
            prediction = np.argmax(probas)
            new_context = []
            for j in range(1, CONTEXT_LENGTH):
                new_context.append(current_context[j])
            sparse_label = np.zeros(self.output_size)
            sparse_label[prediction] = 1
            new_context.append(sparse_label)
            current_context = new_context
            # current_context = Dataset.sparsify(current_context, self.output_size)
            predictions.append(self.voc.token_lookup[prediction])
            if self.voc.token_lookup[prediction] == END_TOKEN:
                break

        return predictions