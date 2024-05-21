import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import gc

# Limite de lignes à charger par étape
BATCH_SIZE = 1200

def load_lines(file_path):
    lines = {}
    with open(file_path, 'r', encoding='iso-8859-1') as file:
        for line in file:
            parts = line.split(" +++$+++ ")
            lines[parts[0]] = parts[-1].strip()
    return lines

def load_conversations(file_path, lines):
    with open(file_path, 'r', encoding='iso-8859-1') as file:
        for line in file:
            parts = line.split(" +++$+++ ")
            line_ids = eval(parts[-1])
            conversation = [lines[line_id] for line_id in line_ids if line_id in lines]
            yield conversation

# Chargez les lignes
lines_file_path = 'movie_lines.txt'
lines = load_lines(lines_file_path)

# Initialiser le tokenizer
tokenizer = Tokenizer(filters='')

# Tokeniser toutes les données pour définir vocab_size avant la construction du modèle
all_texts = list(lines.values())
tokenizer.fit_on_texts(all_texts)
vocab_size = len(tokenizer.word_index) + 1

# Construction du modèle
def create_model(vocab_size):
    model = Sequential([
        Input(shape=(None,)),
        Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True),
        LSTM(256, return_sequences=True),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model(vocab_size)
model.summary()

# Déplacer le max_sequence_length en dehors de la boucle for
max_sequence_length = 0

# Entraînement du modèle par étapes
for epoch in range(10):  # Nombre d'époques d'entraînement
    print(f"Epoch {epoch+1}/10")
    input_texts = []
    target_texts = []
    batch_count = 0

    conversation_generator = load_conversations('movie_conversations.txt', lines)
    
    for conversation in conversation_generator:
        for i in range(len(conversation) - 1):
            input_texts.append(conversation[i])
            target_texts.append(conversation[i + 1])

        # Si le batch est complet, entraîner le modèle
        if len(input_texts) >= BATCH_SIZE:
            batch_count += 1
            print(f"Training batch {batch_count}")

            # Tokenisation et séquençage des textes
            input_sequences = tokenizer.texts_to_sequences(input_texts)
            target_sequences = tokenizer.texts_to_sequences(target_texts)

            # Mise à jour du max_sequence_length
            max_sequence_length = max(max_sequence_length, max([len(seq) for seq in input_sequences + target_sequences]))

            # Padding des séquences
            input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
            target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')

            # Conversion des cibles en format one-hot
            target_sequences = tf.keras.utils.to_categorical(target_sequences, num_classes=vocab_size)

            # Entraînement du modèle avec le mini-batch actuel
            model.fit(input_sequences, target_sequences, batch_size=32, epochs=1)

            # Sauvegarder le modèle
            model.save('chatbot_model.h5')
            del model  # Supprimer le modèle de la mémoire
            tf.keras.backend.clear_session()
            gc.collect()  # Forcer la collecte des objets non utilisés

            # Recharger le modèle
            model = load_model('chatbot_model.h5')

            # Réinitialiser les listes pour le prochain batch
            input_texts = []
            target_texts = []

# Sauvegarder le modèle final
model.save('chatbot_model_final.h5')

# Sauvegarder le tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Fonction pour générer une réponse
def generate_response(question, max_length=20, temperature=1.0):
    global max_sequence_length  # Rendre max_sequence_length global
    # Tokenizer la question
    question_seq = tokenizer.texts_to_sequences([question])
    question_seq = pad_sequences(question_seq, maxlen=max_sequence_length, padding='post')

    # Prédire la réponse
    response = ''
    for _ in range(max_length):
        prediction = model.predict(question_seq, verbose=0)
        predicted_index = sample_with_temperature(prediction[0][-1], temperature)

        # Convertir l'index prédit en mot
        predicted_word = tokenizer.index_word.get(predicted_index, '')

        if predicted_word == '<end>':  # Si la fin de la séquence est prédite
            break

        response += predicted_word + ' '

        # Mettre à jour la séquence d'entrée avec le nouveau mot prédit
        question_seq = np.roll(question_seq, -1, axis=1)
        question_seq[0, -1] = predicted_index

    return response.strip()

def sample_with_temperature(prediction, temperature=1.0):
    prediction = np.squeeze(prediction)  # S'assurer que la prédiction est 1D
    prediction = np.exp(np.log(prediction) / temperature)
    probabilities = prediction / np.sum(prediction)
    return np.random.choice(len(prediction), p=probabilities)

# Exemple d'utilisation
question = "What is your favorite color?"
response = generate_response(question)
print("Chatbot:", response)
