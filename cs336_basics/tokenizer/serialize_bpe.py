import pickle

# stored in .pkl

def save_pkl(saved:object,name:str,path:str):
    with open(path, "wb") as f:
        pickle.dump({name: saved,}, f)

def load_pkl(src:str,name:str):
    with open(src,'rb') as f:
        payload = pickle.load(f)
    return payload.get(name,None)