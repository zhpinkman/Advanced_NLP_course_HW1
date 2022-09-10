import string

alphabet = string.ascii_lowercase

def decrypt(encrypted_message, key):    
    decrypted_message = ""

    for c in encrypted_message:

        if c in alphabet:
            position = alphabet.find(c)
            new_position = (position - key) % 26
            new_character = alphabet[new_position]
            decrypted_message += new_character
        else:
            decrypted_message += c

    words = decrypted_message.split()
    words = [word[::-1] for word in words]
    return ' '.join(words)