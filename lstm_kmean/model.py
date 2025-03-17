import tensorflow as tf
from tensorflow.keras import Model, models, layers, regularizers
import tensorflow_addons as tfa

weight_decay = 1e-4

def enc_conv_block(filters, kernel, strides, padding, rate):
	return models.Sequential([
			layers.Conv1D(filters=filters, kernel_size=kernel, strides=strides, padding=padding),
			layers.Activation(activation='leaky_relu'),
			layers.BatchNormalization(),
			layers.Dropout(rate=rate)
		])


class TripleNet(Model):
	def __init__(self, n_classes=10, n_features=128):
		super(TripleNet, self).__init__()
		# filters   = [ 8,  16,  n_features]
		# ret_seq   = [ True, True, False]
		filters   = [ 32,  n_features]
		ret_seq   = [ True, False]
		# strides = [ 1,   2,  2,   2,]
		# kernel  = [ 7,   7,  3,   3,]
		# padding = ['same', 'same', 'same', 'same']
		self.enc_depth  = len(filters)
		# self.encoder    = [enc_conv_block(filters[idx], kernel[idx], strides[idx], padding[idx], rate=0.1) for idx in range(self.enc_depth)]
		self.encoder   = [layers.LSTM(units=filters[idx], return_sequences=ret_seq[idx]) for idx in range(self.enc_depth)]
		self.flat      = layers.Flatten()
		self.w_1       = layers.Dense(units=n_features, activation='leaky_relu')
		self.w_2       = layers.Dense(units=n_features)
		self.projection_layer = layers.Dense(units=1024, activation="linear")
		self.reshape_layer = layers.Reshape((4, 256))
		# self.feat_norm  = layers.BatchNormalization()
		# self.cls_layer  = layers.Dense(units=n_classes, kernel_regularizer=regularizers.l2(weight_decay))

	def call(self, x):
		for idx in range(self.enc_depth):
			x = self.encoder[idx]( x )
		# print(x.shape)
		x = feat = self.flat( x )
		projected_feat = self.projection_layer(feat)
		reshaped_feat = self.reshape_layer(projected_feat) 
		normalized_feat = tf.nn.l2_normalize(reshaped_feat, axis=-1)
		# print(x.shape)
		# x = feat = self.feat_layer( x )
		# print(x.shape)
		# x = self.feat_norm( x )
		# x = self.cls_layer(x)
		# x = self.w_2( self.w_1( x ) )
		# x = tf.nn.l2_normalize(x, axis=-1)

		return normalized_feat, feat

@tf.function
def train_step(model, optimizer, X, Y):
    with tf.GradientTape() as tape:
        Y_emb, _ = model(X, training=True)

        Y_emb = tf.reshape(Y_emb, [tf.shape(Y_emb)[0], -1])  # Convert (batch, 4, 256) → (batch, 1024)

        if len(Y.shape) > 1 and tf.shape(Y)[1] > 1:
            Y = tf.argmax(Y, axis=1)  # Convert (batch, num_classes) → (batch,)

        loss = tfa.losses.TripletSemiHardLoss()(Y, Y_emb)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def test_step(model, X, Y):
    Y_emb, _ = model(X, training=False)

    Y_emb = tf.reshape(Y_emb, [tf.shape(Y_emb)[0], -1])  # (batch, 4, 256) → (batch, 1024)

    if len(Y.shape) > 1 and tf.shape(Y)[1] > 1:
        Y = tf.argmax(Y, axis=1)  # Convert (batch, num_classes) → (batch,)

    loss = tfa.losses.TripletSemiHardLoss()(Y, Y_emb)
    return loss


