import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = '192.168.43.34'

port = 8080

s.connect((host,port))

tm = s.recv(1024)

s.close()

print("time from server is %s" % tm.decode('ascii'))
