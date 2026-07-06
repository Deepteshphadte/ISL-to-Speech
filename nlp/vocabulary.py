VOCAB = [

    "<PAD>",
    "<UNK>",

    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",

    "1","2","3","4","5","6","7","8","9",

    "HELLO",
    "GOOD",
    "BAD",
    "YES",
    "NO",
    "PLEASE",
    "THANK",
    "YOU",
    "SORRY",
    "I",
    "LOVE"

]

word2idx = {
    word: idx
    for idx, word in enumerate(VOCAB)
}

idx2word = {
    idx: word
    for idx, word in enumerate(VOCAB)
}