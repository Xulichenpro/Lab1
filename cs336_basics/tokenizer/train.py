import regex as re

from pathlib import Path
from multiprocessing import Pool

from .bpe import updated_stats,merge
from .pretokenizer import find_chunk_boundaries

FILE_PATH = Path(__file__).parent.parent / "data/TinyStoriesV2-GPT4-train.txt"
SPECIAL_TOKENS = ["<|endoftext|>"]


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
LEVEL = 255

def train_bpe(
        input_path:str,
        vocab_size:int,
        special_tokens:list[str],
        num_processes = 10,
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    assert vocab_size > LEVEL,"unvalid vocab size"

    chunks = []
    global_stats = {}
    global_cache = {}
    token2bytes = {num:num.to_bytes(1,'big') for num in range(LEVEL + 1)}
    #print(token2bytes)
    bytes2token = {v:k for k,v in token2bytes.items()}
    merges = []

    with open(input_path,'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            #print(chunk)
            #print(_split_by_special_tokens(chunk,SPECIAL_TOKENS))
            chunks.extend(_split_by_special_tokens(chunk,special_tokens))
    #print(chunks)
    with Pool(len(chunks)) as pool:
        stats_cache_list = pool.starmap(_pretokenize,[(chunk,) for chunk in chunks])
    stats_cache_list = [(stats,cache) for stats,cache in stats_cache_list if stats]

    for stats,cache in stats_cache_list:
        for pair,cnt in stats.items():
            global_stats[pair] = global_stats.get(pair,0) + cnt
        for pair,index in cache.items():
            global_cache[pair] = global_cache.get(pair,{})
            for subpair,cnt in index.items():
                global_cache[pair][subpair] = global_cache[pair].get(subpair,0) + cnt
    
    #     print("---------------")
    #     print(f"stats:{stats}")
    #     print(f"cache:{cache}")
    # print(global_stats)
    # print(global_cache)

    for token_id in range(LEVEL + 1,vocab_size + 1):
        merged_pair = max(
            global_stats,
            key = lambda pair : (
                global_stats[pair],
                (token2bytes[pair[0]]).decode('utf-8',errors='replace'),
                (token2bytes[pair[1]]).decode('utf-8',errors='replace'),
            )
        )
        new_bytes = token2bytes[merged_pair[0]] + token2bytes[merged_pair[1]]
        token2bytes[token_id] = new_bytes
        bytes2token[new_bytes] = token_id
        merges.append((token2bytes[merged_pair[0]],token2bytes[merged_pair[1]]))
        index = global_cache[merged_pair]
        #print(list(merged_pair))

        for bytes_stream,cnt in index.items():
            old_bytes_stream = list(bytes_stream)
            bytes_stream = merge(old_bytes_stream,merged_pair,token_id)
            for i,id in enumerate(bytes_stream):
                if token_id != id : continue
                if i >= 1 :
                    global_stats[(bytes_stream[i - 1],token_id)] = cnt + global_stats.get((bytes_stream[i - 1],token_id),0)
                    global_cache[(bytes_stream[i - 1],token_id)] = global_cache.get((bytes_stream[i - 1],token_id),{})
                    global_cache[(bytes_stream[i - 1],token_id)][tuple(bytes_stream)] = global_cache[(bytes_stream[i - 1],token_id)].get(tuple(bytes_stream),0) + cnt
                
                    global_stats[(bytes_stream[i - 1],merged_pair[0])] -= cnt
                    global_cache[(bytes_stream[i - 1],merged_pair[0])][tuple(old_bytes_stream)] -=cnt
                if i <= len(bytes_stream) - 2:
                    global_stats[(token_id,bytes_stream[i + 1])] = cnt + global_stats.get((token_id,bytes_stream[i + 1]),0)
                    global_cache[(token_id,bytes_stream[i + 1])] = global_cache.get((token_id,bytes_stream[i + 1]),{})
                    global_cache[(token_id,bytes_stream[i + 1])][tuple(bytes_stream)] = global_cache[(token_id,bytes_stream[i + 1])].get(tuple(bytes_stream),0) + cnt
                
                    global_stats[(merged_pair[1],bytes_stream[i + 1])] -= cnt
                    global_cache[(merged_pair[1],bytes_stream[i + 1])][tuple(old_bytes_stream)] -=cnt
        del global_stats[merged_pair]
        del global_cache[merged_pair]       
        #break
    
    return token2bytes,merges

def _pretokenize(chunk:str):
    stats = {}
    cache = {}
    match_iter =  re.finditer(PAT,chunk)
    for match in match_iter:
        #print(match.group())
        stats,cache = updated_stats(stats,list(str(match.group()).encode('utf-8')),cache=cache)
    return stats,cache

def _split_by_special_tokens(text:str,special_tokens:list[str] = None) -> list[str]:
    if not special_tokens:
        return [text]
    pattern = '(' + '|'.join(re.escape(s) for s in special_tokens) + ')'
    texts = re.split(pattern, text)
    texts = [t for t in texts if t and t not in special_tokens]
    return texts

def main():
    #chunk = 'aabbccddeeffgg aa'
    #print(_pretokenize(chunk))
    token2bytes,merges = train_bpe(FILE_PATH,100 + LEVEL,SPECIAL_TOKENS)

if __name__ == "__main__":
    main()