class Config:
    BATCH_SIZE = 64 
    img_rows = 28
    img_cols = 28
    channels =1
    z_dim = 100
    img_shape = (img_rows,img_cols,channels)
    EPOCHS = 10
    BATCH_SIZE_TEST = 1000
    SAMPLE_INTERVAL = 1000
    iterations = 20000
    LR = 0.0002
