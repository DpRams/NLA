def findPortAvailable():
    import random
    randomPort = random.randint(8001, 9000)
    if __is_port_in_use(randomPort):
        findPortAvailable()
    else:
        return randomPort

def __is_port_in_use(port: int) -> bool:
    # return True : in use ; return False : not using
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
