def encodeb64(s):
    import base64
    s_bytes = s.encode("utf-8")
    s_base64 = base64.b64encode(s_bytes)
    return s_base64.decode("utf-8")

def decodeb64(scode):
    import base64
    s_bytes = base64.b64decode(scode)
    return s_bytes.decode("utf-8")

def myencodeb64(s, secret_key = ''):
    import random
    level1 = encodeb64(s)
    lengthstring = len(level1)
    random.seed()
    position_insert = random.randint(0, lengthstring)
    temp_string_level1 = level1[:position_insert] + secret_key + level1[position_insert:]
    level2 = encodeb64(temp_string_level1)
    return level2

def mydecodeb64(scode, secret_key = ''):
    level2 = decodeb64(scode)
    temp_string_level1 = level2.replace(secret_key,'')
    level1 = decodeb64(temp_string_level1)
    return level1