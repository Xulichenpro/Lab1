from pathlib import Path

def updated_stats(stats:dict[tuple,int],bytes:list) -> dict[tuple,int]:
    for px,py in zip(bytes,bytes[1:]):
        stats[(px,py)] = stats.get((px,py),0) + 1
    return stats

def merge(bytes:list,merge_id:tuple,new_id:int) -> list:
    new_bytes = []

    i = 0
    while i < len(bytes):
        if i < len(bytes) - 1 and bytes[i] == merge_id[0] and bytes[i + 1] == merge_id[1] :
            new_bytes.append(new_id)
            i += 2
        else:
            new_bytes.append(bytes[i])
            i += 1
    
    return new_bytes

def get_max_key(stats:dict[tuple,int]) -> tuple:
    pair = max(
        stats,
        key = lambda p : (stats[p],p)
    )
    return pair

def main():
    text_path = Path(__file__).parent / "test.txt"
    with open(text_path,'r') as f:
        text = f.read()

    print(f"raw text:{text}")
    bytes = list(text.encode('utf-8'))
    print(f"byte stream:{bytes}")
    stats = updated_stats({},bytes)
    print(f"stats:{stats}")

    print(get_max_key(stats))
    print(f"final : {merge(bytes,get_max_key(stats),256)}")

if __name__ == "__main__":
    main()