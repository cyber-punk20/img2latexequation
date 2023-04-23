import os
import numpy as np
import tensorflow as tf
from preprocessing import *
from Generator import *
from pix2equation import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate(model, df, input_shape, max_output_length, img_npz_path=IMG_NPZ_DIR, verbose=False):
    predictions = []
    ground_truths = []
    # Load the tokenizer
    tokenizer = Dataset()
    tokenizer.voc.loadVolcabulary()
    tokenizer.voc.create_binary_representation()
    i = 0
    blue_scores = []
    smoothing_function = SmoothingFunction().method4
    for index, row in df.iterrows():
        # Preprocess the image
        image = row['image']
        image_name = image[:image.find(".png")]
        if os.path.isfile("{}/{}.npz".format(img_npz_path, image_name)):
            img = np.load("{}/{}.npz".format(img_npz_path, image_name))["features"]
            pad_width = ((0, input_shape[0] - img.shape[0]), (0, 0), (0, 0))
            img = np.pad(img, pad_width, mode='constant')
        else:
            continue

        # Predict the equation
        # img = np.nan_to_num(img, copy=False, nan=0.0, posinf=None, neginf=None)
        img = np.array([img])
        # print(img.shape) # (1,512,128,3)
        # print(img.dtype) # float32 
        # print("Contains NaN:", np.isnan(img).any())
        # print("Contains Inf:", np.isinf(img).any())
        predicted_equation = model.predict_greedy(img, max_output_length)
        ground_truth = Dataset.squashed_seq_to_token_list(row['squashed_seq'], tokenizer.voc)
        # Store the prediction and ground truth for BLEU calculation
        # predictions.append(predicted_equation)
        # ground_truths.append([ground_truth])
        blue_score = sentence_bleu([ground_truth], predicted_equation, smoothing_function=smoothing_function)
        blue_scores.append(blue_score)
        if verbose:
            print('prediction:', ' '.join(predicted_equation))
            print('ground_truth:', ' '.join(ground_truth))
            print('blue_score:', blue_score)
        i += 1
        if i % 100 == 0:
            print(f'evaluate {i} of {df.size}')
    # Calculate the BLEU score
    
    # bleu_scores = [sentence_bleu([gt], pred, smoothing_function=smoothing_function)
    #                for gt, pred in zip(ground_truths, predictions)]
    

    return bleu_scores

def run(output_path=MODEL_DIR, 
        checkpoint_path=CHECKPOINT_PATH,
        dataset_info_path=TEST_DATASET_INFO_PATH,
        max_output_length=MAX_OUTPUT_LENGTH):
    # Load the dataset
    df_test = loadData('my_df_test.pkl')

    # Load the model
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)
    input_shape = tuple(dataset_info["input_shape"])
    output_size = dataset_info["output_size"]
    # steps_per_epoch = int(dataset_info["size"] / BATCH_SIZE)
    # steps_per_epoch = 5
    strategy = tf.distribute.MirroredStrategy()
    model = pix2equation(input_shape, output_size, output_path, checkpoint_path, strategy)
    model.load_for_evaluation()
    # model.compile() # Used for evaluation update
    
    print(df_test)

    # Evaluate the model
    bleu_scores = evaluate(model, df_test, input_shape, max_output_length, verbose=True)
    bleu_scores_arr = np.array(blue_scores)
    with open(EVAL_BLUE_SCORE_PATH, 'w') as f:
        np.savetxt(f, bleu_scores_arr)
    bleu_score = np.mean(bleu_scores)
    print(f"BLEU score: {bleu_score}")

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    tf.debugging.set_log_device_placement(False)
    tf.config.list_physical_devices('GPU')
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available")
        print(tf.config.list_physical_devices('GPU'))
    else:
        print("GPU is not available")
    # tf.config.list_physical_devices('CPU')
    # if tf.config.list_physical_devices('CPU'):
    #     print("CPU is available")
    #     print(tf.config.list_physical_devices('CPU'))
    # else:
    #     print("CPU is not available")
    run(CHECKPOINT_PATH)