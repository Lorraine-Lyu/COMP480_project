import os

import trainer
from encoder import model

samples = trainer.get_most_freq_words()
print("finished preparing test set")
trainer.train_model(model)
print("finished training autoencoder")
print("========testing fp rate of bloomfilter built with encoder===========")
for j in range(12, 28):
    print("test bit array size pow(2,", j, ")")
    trainer.test_collision_rate(model, samples, pow(2, j))

print("==========testing LSH property of hash function==========")

for i in range(10, 20):
    print("test LSH property pow(2,", i, ")")
    trainer.test_LSH_property(model, samples, pow(2, i))

# os.mkdir("./model1")
# model.save("./model1")

# ============= use the encoder as the hash function for LSH ==========
