

LEVEL = 255

def byte_decoder(token2bytes:dict[int,tuple]) -> dict[int,list] :
    token2pair = {num:[num] for num in range(LEVEL + 1)}
   
    for key,(px,py) in token2bytes.items():       
        token2pair[key] = token2pair[px] + token2pair[py]
       
    return token2pair



def main():
    token2bytes = {
        256: (48,49),
        257: (49,50),
        258: (256,23),
        259: (24,257),
        260: (258,259),
    }

    print(byte_decoder(token2bytes))
   
if __name__ == "__main__":
    main()    