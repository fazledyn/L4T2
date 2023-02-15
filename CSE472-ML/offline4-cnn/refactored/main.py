from layers import *
from model import *
from utils import *
import sys


def main_test():

    cnn = CNN()
    cnn = pickle.load(open('1705066_model.pickle', 'rb'))
    print("CNN Loaded from Pickle File!")

    input_folder = sys.argv[1]
    input_files = os.listdir(input_folder)

    exts = [ "png", "jpg", "jpeg", "PNG", "JPG", "JPEG" ]
    input_files = [ file for file in input_files if file.split(".")[-1] in exts ]

    BATCH_SIZE = 32
    print(f"Prediction will be done in batches of {BATCH_SIZE}")
    n_batch = int(np.ceil(len(input_files)/BATCH_SIZE))

    y_pred = []
    for i in tqdm(range(n_batch)):
        X_batch = input_files[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        X_batch = np.array([ load_image_as_grayscale(f"{input_folder}/{filename}") for filename in X_batch ])
        y_batch = cnn.predict(X_batch)
        y_pred.extend(np.argmax(y_batch, axis=1))

    df = pd.DataFrame()
    df["FileName"] = input_files
    df["Digit"] = y_pred

    df.to_csv(f"{input_folder}/1705066_prediction.csv", index=False)
    print("CSV File Generated!")


def compare_testset(testfile, predfile):
    
        test_df = pd.read_csv(testfile)
        pred_df = pd.read_csv(predfile)
    
        test_df = test_df.sort_values(by=['filename'])
        pred_df = pred_df.sort_values(by=['FileName'])
    
        test_df = test_df.reset_index(drop=True)
        pred_df = pred_df.reset_index(drop=True)

        y_true = test_df['digit'].values
        y_pred = pred_df['Digit'].values

        print("Accuracy: ", accuracy_score(y_true, y_pred))
        print("F1 Score: ", f1_score(y_true, y_pred, average='macro'))

        conf_mat = confusion_matrix(y_true, y_pred)
        conf_dis = ConfusionMatrixDisplay(conf_mat)
        conf_dis.plot()
        plt.savefig('confusion_matrix.png')

        print("Confusion Matrix Saved!")



def main_train():
    
    cnn = CNN(learning_rate=LEARNING_RATE)

    cnn.add(ConvolutionLayer(n_filter=6, n_channel_in=1, kernel_size=5, stride=1, padding=0))
    cnn.add(ReLUActivationLayer())
    cnn.add(MaxPoolingLayer(pool_size=2, stride=2))

    cnn.add(ConvolutionLayer(n_filter=16, n_channel_in=6, kernel_size=5, stride=1, padding=0))
    cnn.add(ReLUActivationLayer())
    cnn.add(MaxPoolingLayer(pool_size=2, stride=2))

    cnn.add(FlatteningLayer())
    
    cnn.add(DenseLayer(n_output=84))
    cnn.add(DenseLayer(n_output=10))

    cnn.add(SoftMaxLayer())
    
    X_train, _, y_train, _ = load_dataset(test_size=0.97, shuffle=True)
    _, X_valid, _, y_valid = load_dataset(test_size=0.01, shuffle=True)
    
    print("Dataset Loaded:", DATASET_NAME)
    print("Train Data Size:", X_train.shape)
    print("Valid Data Size:", X_valid.shape)

    cnn.train(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        batch_size=32,
        epochs=10
    )

    print("Training Complete!")
    save_model(cnn, f"{REPORT_FOLDER}/model_lr{LEARNING_RATE}.pkl")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: python test_1705066.py <test_file>')
        exit(0)

    """
    Use specfic function for testing or training
    """
    # main_test()
    # main_train()
    # compare_testset("training-d.csv", "1705066_prediction.csv")