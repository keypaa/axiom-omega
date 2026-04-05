import tiktoken

def count_token_in_file(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        text = f.read()
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)

    return len(tokens)

file_name = "Claude-Master plan validation and implementation.md"
toke_count = count_token_in_file(file_name)

print(f"Token Count : {toke_count}")
