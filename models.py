import tensorflow as tf
from tensorflow.keras import layers, models
from metrics import dice_coef, dice_loss, iou_metric

class TumorSegmentationModel:
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.model = self.build_advanced_unet()

    def build_advanced_unet(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        # Encoder Path
        def encoder_block(x, filters, dropout_rate=0.2):
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate)(x)
            p = layers.MaxPooling2D((2, 2))(x)
            return x, p



        # Decoder Path
        def decoder_block(x, skip, filters, dropout_rate=0.1):
            x = layers.UpSampling2D((2, 2))(x)
            x = layers.concatenate([x, skip])
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate)(x)
            return x

        # Encoder
        c1, p1 = encoder_block(inputs, 32)
        c2, p2 = encoder_block(p1, 64)
        c3, p3 = encoder_block(p2, 128)
        c4, p4 = encoder_block(p3, 256)

        # Bottleneck
        b = layers.Conv2D(512, (3, 3), padding='same')(p4)
        b = layers.BatchNormalization()(b)
        b = layers.Activation('relu')(b)
        b = layers.Conv2D(512, (3, 3), padding='same')(b)
        b = layers.BatchNormalization()(b)
        b = layers.Activation('relu')(b)

        # Decoder
        d1 = decoder_block(b, c4, 256)
        d2 = decoder_block(d1, c3, 128)
        d3 = decoder_block(d2, c2, 64)
        d4 = decoder_block(d3, c1, 32)

        # Output
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)

        model = models.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=dice_loss,
                      metrics=[dice_coef, 'accuracy', iou_metric])
        return model