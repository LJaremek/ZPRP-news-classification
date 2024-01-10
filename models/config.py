# for preprocess
PATH_TO_DATA = '/home/wpartycja/7sem/zrpr/zprp-projekt-z4/data/fake_news.csv'
SEQ_LEN = 200
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42

# for LSTM
NUM_CLASSES = 2

BATCH_SIZE = 32
EMBEDDING_DIM = 50
HIDDEN_DIM = 200

LSTM_LAYERS = 1
IS_BIDIRECTIONAL = False

LR = 4e-4
DROPOUT = 0.5

EPOCHS = 10
