'''a conditional generative adversarial network designed specifically for working with cryptopunks art projects'''

class ConditionalGAN(Model):
    def __init__(self):
        self.batch_size = 64
        self.num_channels = 1
        self.num_classes = 10
        self.image_size = 28
        self.latent_dim = 128
        self.learning_rate = 0.0002
        self.beta_1 = 0.5
        self.generator_in_channels = self.latent_dim + self.num_classes
        self.discriminator_in_channels = self.num_channels + self.num_classes
        self.opt = Adam(lr=self.learning_rate, 
                        beta_1 = self.beta_1)
        
    def __setattr__(self, name, value):
        print(name)
        return super().__setattr__(name, value)
    
    # define the standalone discriminator model
    def define_discriminator(self, in_shape=(24, 24, 3)):
        self.discriminator = Sequential()
        # downsample
        self.discriminator.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        # downsample
        self.discriminator.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        # classifier
        self.discriminator.add(Flatten())
        self.discriminator.add(Dropout(0.4))
        self.discriminator.add(Dense(1, activation='sigmoid'))
        
    def define_generator(self):
        self.generator = Sequential()
        # foundation for 6*6 image
        n_nodes = 128 * 6 * 6
        self.generator.add(Dense(n_nodes, 
                                 input_dim=self.latent_dim))
        self.generator.add(LeakyReLU(alpha=0.2))
        self.generator.add(Reshape((6, 6, 128)))
        # upsample to 12x12
        self.generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        self.generator.add(LeakyReLU(alpha=0.2))
        # upsample to 24x24
        self.generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        self.generator.add(LeakyReLU(alpha=0.2))
        # generate
        self.generator.add(Conv2D(3, (6, 6), activation='tanh', padding='same'))

        
    def define_gan(generator, discriminator):
        # make weights in the discriminator not trainable
        discriminator.trainable = False
        # connect them
        self.model = Sequential()
        # add generator
        self.model.add(self.generator)
        # add the discriminator
        self.model.add(discriminator)
        
    def generate_real_samples(self):
        # choose random instances
        ix = randint(0, self.dataset.shape[0], self.n_samples)
        # select images
        X = self.dataset[ix]
        # generate class labels
        y = ones((self.n_samples, 1))
        return X, y
    
    def generate_latent_points(self):
        'generate points in latent space as input for the generator'
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input
    
    def generate_fake_samples(self):
        # generate points in latent space
        x_input = generate_latent_points(latent_dim, n_samples)
        # predict outputs
        X = generator.predict(x_input)
        # create class labels
        y = zeros((n_samples, 1))
        return X, y

    def train(n_epochs=1):
        '''train the generator and discriminator'''
        bat_per_epo = int(dataset.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                X_real, y_real = generate_real_samples(dataset, half_batch)
                # update discriminator model weights
                d_loss1, _ = d_model.train_on_batch(X_real, y_real)
                # generate 'fake' examples
                X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
                # update discriminator model weights
                d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
                # prepare points in latent space as input for the generator
                X_gan = generate_latent_points(latent_dim, n_batch)
                # create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                    (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # save the generator model
        g_model.save('generator.h5')
        
    def compile(self, opt):
        # compile model
        opt = Adam(lr=0.0002, 
                   beta_1=0.5)
        self.discriminator.compile(loss='binary_crossentropy', 
                                   optimizer=opt, 
                                   metrics=['accuracy'])
        self.model.compile(loss='binary_crossentropy', 
                           optimizer=opt)
        
    def predict(self):
     return True
