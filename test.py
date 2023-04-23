from preprocessing import *
from constants import *

def show_img(image_name, img_npz_path=IMG_NPZ_DIR):
    if os.path.isfile("{}/{}.npz".format(img_npz_path, image_name)):
        img = np.load("{}/{}.npz".format(img_npz_path, image_name))["features"]
        img = img * 255
        img = np.array(img, dtype=np.uint8)
        Dataset.show(img)


if __name__ == '__main__':
    # show_img('00a9a9b4af8cc4a_basic')
    dataset = Dataset()
    dataset.voc.loadVolcabulary()
    dataset.voc.create_binary_representation()
    print(dataset.voc.vocabulary)
    df = loadData('df_test.pkl')
    for index, row in df.iterrows():
        equ_token_id_seq = row['squashed_seq']
        equ_token_id_seq = [x for x in equ_token_id_seq if x != dataset.voc.vocabulary[END_TOKEN]]
        token_id_sequence = [START_TOKEN]
        token_id_sequence.extend([dataset.voc.token_lookup[id] for id in equ_token_id_seq])
        token_id_sequence.append(END_TOKEN)
        print(' '.join(token_id_sequence))