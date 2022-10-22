def is_port_in_use(port: int) -> bool:
    # return True : in use ; return False : not using
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

print(is_port_in_use(8002))