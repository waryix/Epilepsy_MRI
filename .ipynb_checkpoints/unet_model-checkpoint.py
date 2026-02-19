import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# =========================
# Dice Coefficient Metric
# =========================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# =========================
# Dice Loss
# =========================
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# =========================
# Improved U-Net
# =========================
def unet(input_size=(256, 256, 1), dropout_rate=0.5):
    inputs = Input(input_size)
    
    # --------- Encoder ---------
    def conv_block(x, filters):
        x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        return x

    c1 = conv_block(inputs, 32)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = conv_block(p1, 64)
    p2 = MaxPooling2D((2,2))(c2)

    c3 = conv_block(p2, 128)
    p3 = MaxPooling2D((2,2))(c3)

    c4 = conv_block(p3, 256)
    d = Dropout(dropout_rate)(c4)

    # --------- Decoder ---------
    def decoder_block(x, skip, filters):
        x = Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(x)
        x = concatenate([x, skip])
        x = conv_block(x, filters)
        x = Dropout(dropout_rate/2)(x)  # optional dropout in decoder
        return x

    u5 = decoder_block(d, c3, 128)
    u6 = decoder_block(u5, c2, 64)
    u7 = decoder_block(u6, c1, 32)

    outputs = Conv2D(1, (1,1), activation='sigmoid')(u7)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=[dice_coef])

    return model

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    model = unet()
    model.summary()
